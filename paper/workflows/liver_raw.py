"""Public bulk/spatial count archives -> HSC decomposition and spatial gene scores.

No stored gene effect, signature score or section score is used as an input.
The in_frozen_axis flag and the two original identity-gate scores require separate
upstream analyses and are deliberately absent from the corresponding outputs.
"""
import argparse,csv,gzip,io,json,re,tarfile
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def sample_groups(matrix):
    fields={}
    with gzip.open(matrix,'rt') as f:
        for line in f:
            if line.startswith(('!Sample_geo_accession\t','!Sample_title\t')):
                parts=next(csv.reader([line],delimiter='\t'));fields[parts[0]]=parts[1:]
    titles=dict(zip(fields['!Sample_geo_accession'],fields['!Sample_title']))
    if len(titles)!=12:raise ValueError('expected 12 public bulk samples')
    groups={k:[] for k in ['pH_S','pH_T','LX_S','LX_T']}
    for gsm,title in titles.items():
        key=('pH' if title.upper().startswith('PHSC') else 'LX')+('_T' if 'TGF' in title.upper() else '_S')
        groups[key].append(gsm)
    if any(len(x)!=3 for x in groups.values()):raise ValueError('expected 3 biological samples in each group')
    return groups


def regions(directory):
    rows=[]
    for group in ['BA1','BA2','NL','NonBA']:
        with gzip.open(directory/f'GSE338525_{group}_metadata.csv.gz','rt') as f:
            reader=csv.reader(f);next(reader)
            for row in reader:
                if len(row)!=9:raise ValueError('official metadata requires nine fields including the unlabeled cluster field')
                full,section,spot,duplicate,cluster,hep,stellate,chol,region=row
                if full!=duplicate or full!=f'{section}_{spot}':raise ValueError('inconsistent spot identities')
                if section=='BA1_3':raise ValueError('author-excluded section appeared in metadata')
                rows.append(dict(section=section,barcode=spot,region=region if region!='None' else np.nan,cell2loc_hep_prop=float(hep),cell2loc_stellate_prop=float(stellate),cell2loc_chol_prop=float(chol)))
    out=pd.DataFrame(rows)
    if out.duplicated(['section','barcode']).any():raise ValueError('duplicate metadata spot keys')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bulk-tar',type=Path,required=True);p.add_argument('--spatial-tar',type=Path,required=True);p.add_argument('--metadata-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    groups=sample_groups(a.metadata_dir/'GSE253493_series_matrix.txt.gz');counts={}
    with tarfile.open(a.bulk_tar) as archive:
        for member in sorted(archive.getmembers(),key=lambda x:x.name):
            if not member.name.endswith('.txt.gz'):continue
            gsm=re.search(r'GSM\d+',member.name).group()
            with gzip.open(archive.extractfile(member),'rt') as f:d=pd.read_csv(f,sep='\t',skiprows=1,usecols=[0,6],index_col=0)
            counts[gsm]=d.iloc[:,0]
    X=pd.DataFrame(counts);X.index=X.index.str.split('.').str[0];X=X.groupby(level=0).sum()
    if X.isna().any().any() or (X<0).any().any() or (X.sum(0)<=0).any():raise ValueError('invalid count matrix')
    log=np.log2(1+1e6*X/X.sum(0));ph=log[groups['pH_T']].mean(1)-log[groups['pH_S']].mean(1);lx=log[groups['LX_T']].mean(1)-log[groups['LX_S']].mean(1)
    with tarfile.open(a.spatial_tar) as archive:
        names=archive.getnames();h5names=[n for n in names if n.endswith('filtered_feature_bc_matrix.h5')]
        with h5py.File(io.BytesIO(archive.extractfile(sorted(h5names)[0]).read()),'r') as h:
            ids=np.array([x.split('.')[0] for x in h['matrix/features/id'].asstr()[:]]);symbols=h['matrix/features/name'].asstr()[:]
        mapping=dict(zip(ids,symbols));ok=((X>10).sum(1)>=6)&X.index.isin(mapping)
        effects=pd.DataFrame(dict(ensg=X.index[ok],sym=[mapping[g] for g in X.index[ok]],pHSC=ph[ok].to_numpy(),LX2=lx[ok].to_numpy()))
        core=effects[(effects.pHSC>1)&(effects.LX2>1)].copy();core.to_csv(a.output/'liver_core94_signature.csv',index=False)
        response=effects[effects.pHSC>1].copy();response['B']=(response.pHSC+response.LX2)/2;response['Delta']=(response.pHSC-response.LX2)/2
        response['dominant']=np.where(response.B.abs()>response.Delta.abs(),'B (shared)','Delta (pHSC-specific residual)')
        response.to_csv(a.output/'liver_adapter_decomposition.csv',index=False)
        sets={'core94':set(core.ensg),'Bdom':set(response[response.dominant.str.startswith('B ')].ensg),'Ddom':set(response[response.dominant.str.startswith('Delta')].ensg)}
        print('raw-derived gene sets',{k:len(v) for k,v in sets.items()},flush=True)
        frames=[]
        for name in h5names:
            section=re.match(r'GSM\d+_(.+)_filtered_feature_bc_matrix\.h5',Path(name).name).group(1)
            posname=[n for n in names if n.startswith(name.split('_filtered')[0]) and n.endswith('tissue_positions_list.csv.gz')][0]
            with h5py.File(io.BytesIO(archive.extractfile(name).read()),'r') as h:
                m=h['matrix'];ids=m['features/id'].asstr()[:];barcodes=m['barcodes'].asstr()[:]
                counts=sparse.csc_matrix((m['data'][:],m['indices'][:],m['indptr'][:]),shape=tuple(m['shape'][:])).T.tocsr()
            totals=np.maximum(np.asarray(counts.sum(1)).ravel(),1)
            pos=pd.read_csv(io.BytesIO(gzip.decompress(archive.extractfile(posname).read())),header=None,names=['barcode','in_tissue','array_row','array_col','pxl_row','pxl_col']).set_index('barcode').reindex(barcodes)
            rec=dict(section=section,barcode=barcodes,x=pos.pxl_col.to_numpy(),y=pos.pxl_row.to_numpy(),umi=totals);cols=counts.tocsc()
            for key,genes in sets.items():
                idx=np.flatnonzero(np.isin(ids,list(genes)))
                if not len(idx):raise ValueError('empty mapped spatial gene set')
                expr=np.log1p(cols[:,idx].toarray()*(1e4/totals)[:,None]);z=(expr-expr.mean(0))/np.maximum(expr.std(0),1e-9);rec[key]=z.mean(1)
            frames.append(pd.DataFrame(rec));print(section,len(barcodes),flush=True)
    spots=pd.concat(frames,ignore_index=True).merge(regions(a.metadata_dir),on=['section','barcode'],how='left',validate='one_to_one').dropna(subset=['x','y'])
    spots=spots[spots.section.isin(regions(a.metadata_dir).section.unique())].copy()
    spots[['x','y']]=spots[['x','y']].astype(int);labeled=spots.dropna(subset=['region']);rows=[]
    for section,d in labeled.groupby('section'):
        row=dict(section=section,group=re.sub(r'_\d+$','',section),n_spots=len(d),n_scar=int((d.region=='Scar').sum()),n_hep=int((d.region=='Hep').sum()),n_chol=int((d.region=='Chol').sum()))
        for key in sets:
            row[key+'_scar']=d.loc[d.region=='Scar',key].mean();row[key+'_hep']=d.loc[d.region=='Hep',key].mean();row[key+'_scar_minus_hep']=row[key+'_scar']-row[key+'_hep']
        rows.append(row)
    section_table=pd.DataFrame(rows).round(6)
    # Historical CSV boundary: scores rounded to six decimals before hep centering.
    props=['cell2loc_hep_prop','cell2loc_stellate_prop','cell2loc_chol_prop']
    spots=spots[['section','barcode','x','y','umi','region','core94','Bdom','Ddom']+props].round(6)
    hep=spots[spots.region=='Hep'].groupby('section')[list(sets)].mean()
    for key in sets:spots[key+'_h']=spots[key]-spots.section.map(hep[key])
    labeled=spots.dropna(subset=['region'])
    for key in sets:section_table[key+'_scar_median_h']=section_table.section.map(labeled[labeled.region=='Scar'].groupby('section')[key+'_h'].median())
    frac={section:float((d.loc[d.region=='Scar','core94_h']>d.loc[d.region=='Hep','core94_h'].median()).mean()) for section,d in labeled.groupby('section')}
    section_table['frac_scar_above_hep_median']=section_table.section.map(frac)
    spots=spots[['section','barcode','x','y','umi','region','core94','Bdom','Ddom','core94_h','Bdom_h','Ddom_h']+props]
    spots.round(6).to_csv(a.output/'liver_spatial_spot_scores.csv',index=False);section_table.round(6).to_csv(a.output/'liver_spatial_section_contrast.csv',index=False)
    (a.output/'liver_scope.json').write_text(json.dumps({'raw_counts':True,'gene_set_sizes':{k:len(v) for k,v in sets.items()},'omitted_signature_columns':['in_frozen_axis'],'omitted_spot_columns':['hsc_identity','signed_hsc_state'],'reason':'Separate frozen-axis and identity-gate provenance not reconstructed; no historical values copied.'},indent=2)+'\n')
if __name__=='__main__':main()
