"""Rebuild the distinct 16-line, 86-condition Tahoe weight-rule reconstruction.

Preserves the archived float32 cell normalization and saved-array boundaries.
It is a different cohort and random seed from the 15-context release experiment.
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from tahoe_raw import SPEC,labels


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--block-cells',type=int,default=100000);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    genes=pd.read_csv(SPEC/'gene-axis.tsv',sep='\t')
    with h5py.File(a.input,'r') as h:
        cl,lc=labels(h,'cell_line');dc,pc=labels(h,'drugname_drugconc');ctl=list(dc).index("[('DMSO_TF', 0.0, 'uM')]")
        table=pd.crosstab(lc,pc);snu=list(cl).index('CVCL_0366');lines=table.sum(axis=1).sort_values(ascending=False).index[:15].tolist()+[snu]
        if len(set(lines))!=16:raise ValueError('expected 15 non-SNU contexts plus SNU-423')
        drugs=[c for c in table.columns if c!=ctl and (table.loc[lines,c]>=400).all()]
        if len(drugs)!=86:raise ValueError('historical 86-condition cohort changed')
        names=h['var/gene_name'].asstr()[:];first={}
        for i,name in enumerate(names):first.setdefault(name,i)
        axis=[g for g in genes.gene if g in first];lut=np.full(len(names),-1,np.int32)
        for i,name in enumerate(axis):lut[first[name]]=i
        nL,nD,nG=16,len(drugs),len(axis);off_c=nL*nD;off_st=off_c+2*nL;off_sc=off_st+nL*nD;ng=off_sc+2*nL
        gd=np.full(len(lc),-1,np.int32);gs=gd.copy();rng=np.random.default_rng(23)
        for i,line in enumerate(lines):
            for j,drug in enumerate(drugs):
                ix=rng.permutation(np.flatnonzero((lc==line)&(pc==drug)))[:400];gd[ix]=i*nD+j;gs[ix[:20]]=off_st+i*nD+j
            ix=rng.permutation(np.flatnonzero((lc==line)&(pc==ctl)))
            for f in (0,1):
                take=ix[f*400:(f+1)*400];gd[take]=off_c+2*i+f;gs[take[:10]]=off_sc+2*i+f
        ip=h['X/indptr'][:];data,indices=h['X/data'],h['X/indices'];sums=[np.zeros((ng,nG)),np.zeros((ng,nG))]
        for start in range(0,len(lc),a.block_cells):
            end=min(start+a.block_cells,len(lc));lo,hi=map(int,[ip[start],ip[end]])
            cell=np.repeat(np.arange(start,end,dtype=np.int32),np.diff(ip[start:end+1]));selected=gd[cell]>=0
            # Retain every nonzero gene of selected cells for their full library sum.
            dd=data[lo:hi].astype(np.float32)[selected];ii=indices[lo:hi][selected];cell=cell[selected]
            library=np.bincount(cell-start,weights=dd.astype(float),minlength=end-start)
            w=np.log1p(1e4*dd/np.maximum(library[cell-start],1).astype(np.float32));ga=lut[ii]
            for group,acc in zip([gd,gs],sums):
                gg=group[cell];mask=(gg>=0)&(ga>=0)
                acc+=np.bincount(gg[mask].astype(np.int64)*nG+ga[mask],weights=w[mask].astype(float),minlength=ng*nG).reshape(ng,nG)
            if start%(5*a.block_cells)==0:print('16-line rows',end,'/',len(lc),flush=True)
        cd=np.bincount(gd[gd>=0],minlength=ng);cs=np.bincount(gs[gs>=0],minlength=ng)
        deep=(sums[0]/np.maximum(cd,1)[:,None]).astype(np.float32);small=(sums[1]/np.maximum(cs,1)[:,None]).astype(np.float32)
        np.savez_compressed(a.output/'plate7_16line_pseudobulk.npz',deep=deep,small=small,cnt_d=cd,cnt_s=cs,lines=cl[lines].astype(str),conds=dc[drugs].astype(str),offsets=np.array([off_c,off_st,off_sc]),genes=np.array(axis))
        print('16-line reconstruction complete',deep.shape,flush=True)
if __name__=='__main__':main()
