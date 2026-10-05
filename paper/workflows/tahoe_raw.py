"""Raw Tahoe counts -> frozen-cohort cell-normalized pseudobulk.

Plate 2 can read the original publisher H5AD or its byte-range CSR chunk cache.
The supplied cohort/gene specs are pre-expression selection metadata, not results.
Plate 7 rebuilds the historical cohort from the H5AD metadata and maps gene names.
"""
from pathlib import Path
import argparse,json,zlib
import h5py
import numpy as np
import pandas as pd

SPEC=Path(__file__).resolve().parents[1]/'specs/tahoe'

def labels(h,name):
    g=h['obs'][name]
    if isinstance(g,h5py.Group):
        cats=np.asarray(g['categories'].asstr()[:]);return cats,g['codes'][:]
    a=np.asarray(g.asstr()[:]);return np.unique(a,return_inverse=True)


def plate2(args):
    contexts=pd.read_csv(SPEC/'contexts.tsv',sep='\t');conditions=pd.read_csv(SPEC/'conditions.tsv',sep='\t')
    genes=pd.read_csv(SPEC/'gene-axis.tsv',sep='\t')
    if args.cache:
        md=args.cache/'metadata-cache-v1';m=json.loads((md/'manifest.json').read_text())
        code={n:np.load(md/f'{n}.codes.int8.npy') for n in ['cell_line','drugname_drugconc','pass_filter']}
        cats={n:m['fields'][n]['categories'] for n in code}
        ip=np.load(args.cache/'x-chunk-cache-v1/X-indptr.int64.npy');handle=None
    else:
        handle=h5py.File(args.input,'r');code={};cats={}
        for n in ['cell_line','drugname_drugconc','pass_filter']:cats[n],code[n]=labels(handle,n)
        ip=handle['X/indptr'][:]
        names=handle['var/gene_name'].asstr()[:]
        if not np.array_equal(np.char.upper(names[genes.tahoe_column].astype(str)),np.char.upper(genes.gene.to_numpy(str))):raise ValueError('gene axis differs from frozen selection')
    selected=[];valid=code['pass_filter']==list(cats['pass_filter']).index('full')
    for row in contexts.itertuples():
        for condition in conditions.itertuples():
            ix=np.flatnonzero(valid & (code['cell_line']==row.cell_line_code) & (code['drugname_drugconc']==condition.condition_code))[:20]
            if len(ix)!=20:raise ValueError('insufficient cells in frozen cohort')
            selected.extend((int(i),row.cell_line,condition.condition) for i in ix)
    meta=pd.DataFrame(sorted(selected),columns=['row','cell_line','condition'])
    rows=meta.row.to_numpy();starts,ends=ip[rows],ip[rows+1];lens=ends-starts;off=np.r_[0,np.cumsum(lens)]
    data=np.empty(off[-1],np.float32);indices=np.empty(off[-1],np.int32)
    if args.cache:
        chunk=96077;needed=set()
        for a,b in zip(starts,ends):needed.update(range(int(a//chunk),int((b-1)//chunk)+1))
        cd=args.cache/'x-chunk-cache-v1/chunks';cur=0
        for ci in sorted(needed):
            s0=ci*chunk
            d=np.frombuffer(zlib.decompress((cd/f'data-{s0:012d}.z').read_bytes()),np.float32)
            ix=np.frombuffer(zlib.decompress((cd/f'indices-{s0:012d}.z').read_bytes()),np.int64)
            while cur<len(rows) and starts[cur]<s0+chunk:
                lo,hi=max(starts[cur],s0),min(ends[cur],s0+chunk)
                if hi>lo:
                    pos=int(off[cur]+lo-starts[cur]);data[pos:pos+hi-lo]=d[lo-s0:hi-s0];indices[pos:pos+hi-lo]=ix[lo-s0:hi-s0]
                if ends[cur]<=s0+chunk:cur+=1
                else:break
        if cur!=len(rows):raise ValueError('incomplete raw chunk coverage')
    else:
        for j,(a,b) in enumerate(zip(starts,ends)):
            data[off[j]:off[j+1]]=handle['X/data'][a:b];indices[off[j]:off[j+1]]=handle['X/indices'][a:b]
        handle.close()
    cell=np.repeat(np.arange(len(rows)),lens);library=np.bincount(cell,weights=data.astype(float),minlength=len(rows))
    if np.any(library<=0):raise ValueError('zero library')
    lut=np.full(62710,-1,np.int32);lut[genes.tahoe_column]=genes.axis_column;g=lut[indices];keep=g>=0
    weights=np.log1p(1e4*data.astype(float)/library[cell]);meta['grp']=meta.groupby(['cell_line','condition']).ngroup()
    gid=meta.grp.to_numpy()[cell];ng=int(meta.grp.max()+1);nG=len(genes)
    key=meta.drop_duplicates('grp').set_index('grp')[['cell_line','condition']].sort_index()
    ctrl=conditions.loc[conditions.is_control==1,'condition'].item();isctrl=(meta.condition==ctrl).to_numpy()
    half=np.zeros(len(meta),np.int8);rng=np.random.default_rng(11)
    for _,ix in meta[isctrl].groupby('grp').indices.items():half[meta[isctrl].index[ix]]=rng.permutation(np.repeat([0,1],10))
    means=[]
    for f in (0,1):
        selected=(~isctrl)|(half==f);mask=selected[cell]&keep
        total=np.bincount(gid[mask]*nG+g[mask],weights=weights[mask],minlength=ng*nG).reshape(ng,nG)
        count=np.bincount(meta.grp.to_numpy()[selected],minlength=ng);means.append(total/count[:,None])
    ctx=contexts.cell_line.to_numpy(str);targets=conditions.loc[conditions.is_control==0,'condition'].to_numpy(str)
    index={(row.cell_line,row.condition):i for i,row in key.iterrows()}
    T=np.array([[means[0][index[c,t]] for t in targets] for c in ctx])
    A=np.array([means[0][index[c,ctrl]] for c in ctx]);B=np.array([means[1][index[c,ctrl]] for c in ctx])
    np.savez_compressed(args.output/'plate2_pseudobulk.npz',treated=T,control_a=A,control_b=B,contexts=ctx,targets=targets,genes=genes.gene.to_numpy(str))
    meta.drop(columns='grp').to_csv(args.output/'plate2_selected_cells.tsv',sep='\t',index=False)
    print('plate2',T.shape,'raw cells',len(rows),flush=True)


def plate7(args):
    genes=pd.read_csv(SPEC/'gene-axis.tsv',sep='\t')
    with h5py.File(args.input,'r') as h:
        cl,lc=labels(h,'cell_line');dc,pc=labels(h,'drugname_drugconc')
        controls=[i for i,s in enumerate(dc) if 'DMSO' in s.upper()]
        table=pd.crosstab(lc,pc);lines=table.sum(1).sort_values(ascending=False).index[:15].tolist()
        drugs=[c for c in table.columns if c not in controls and (table.loc[lines,c]>=400).all()][:88]
        names=h['var/gene_name'].asstr()[:];first={}
        for i,g in enumerate(names):first.setdefault(g,i)
        axis=[g for g in genes.gene if g in first];nG=len(axis);lut=np.full(len(names),-1,np.int32)
        for i,g in enumerate(axis):lut[first[g]]=i
        nL,nD=len(lines),len(drugs);offset_c=nL*nD;offset_st=offset_c+2*nL;offset_sc=offset_st+nL*nD;ng=offset_sc+2*nL
        gd=np.full(len(lc),-1,np.int64);gs=gd.copy();rng=np.random.default_rng(11)
        for i,line in enumerate(lines):
            for j,drug in enumerate(drugs):
                ix=rng.permutation(np.flatnonzero((lc==line)&(pc==drug)))[:400]
                gd[ix]=i*nD+j;gs[ix[:20]]=offset_st+i*nD+j
            ix=rng.permutation(np.flatnonzero((lc==line)&(pc==controls[0])))
            if len(ix)<800:raise ValueError('insufficient controls')
            for f in (0,1):
                take=ix[f*400:(f+1)*400];gd[take]=offset_c+2*i+f;gs[take[:10]]=offset_sc+2*i+f
        ip=h['X/indptr'][:];sums=[np.zeros((ng,nG)),np.zeros((ng,nG))]
        for start in range(0,len(lc),args.block_cells):
            end=min(start+args.block_cells,len(lc))
            if not (gd[start:end]>=0).any():continue
            lo,hi=ip[start],ip[end];data=h['X/data'][lo:hi].astype(float);idx=h['X/indices'][lo:hi]
            cell=np.repeat(np.arange(start,end),np.diff(ip[start:end+1]));lib=np.bincount(cell-start,weights=data,minlength=end-start)
            w=np.log1p(1e4*data/np.maximum(lib[cell-start],1));ga=lut[idx]
            for group,acc in zip([gd,gs],sums):
                v=group[cell];mask=(v>=0)&(ga>=0)
                acc+=np.bincount(v[mask]*nG+ga[mask],weights=w[mask],minlength=ng*nG).reshape(ng,nG)
            if start%(10*args.block_cells)==0:print('plate7 rows',end,'/',len(lc),flush=True)
        outputs={}
        for tag,group,acc,oT,oC in [('deep',gd,sums[0],0,offset_c),('shallow',gs,sums[1],offset_st,offset_sc)]:
            count=np.bincount(group[group>=0],minlength=ng);pb=acc/np.maximum(count,1)[:,None]
            outputs[tag+'_treated']=pb[oT:oT+nL*nD].reshape(nL,nD,nG)
            outputs[tag+'_control_a']=pb[oC:oC+2*nL:2];outputs[tag+'_control_b']=pb[oC+1:oC+2*nL:2]
        np.savez_compressed(args.output/'plate7_pseudobulk.npz',**outputs,contexts=cl[lines].astype(str),targets=dc[drugs].astype(str),genes=np.array(axis))
        pd.DataFrame({'row':np.flatnonzero(gd>=0),'deep_group':gd[gd>=0],'shallow_group':gs[gd>=0]}).to_csv(args.output/'plate7_selected_cells.tsv',sep='\t',index=False)
        print('plate7 complete',nL,nD,nG,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plate',choices=['2','7'],required=True)
    src=p.add_mutually_exclusive_group(required=True);src.add_argument('--input',type=Path);src.add_argument('--cache',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--block-cells',type=int,default=20000);a=p.parse_args()
    if a.block_cells<1:p.error('positive block size required')
    if a.plate=='7' and not a.input:p.error('plate7 requires H5AD')
    a.output.mkdir(parents=True,exist_ok=True)
    (plate2 if a.plate=='2' else plate7)(a)
if __name__=='__main__':main()
