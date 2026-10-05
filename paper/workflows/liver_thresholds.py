"""Recompute section-level threshold sensitivity from counts and official regions.

This reproduces the per-section table used by the active figures. It does not
claim to rerun the separate expression-matched random-gene null experiment.
"""
import argparse,io,re,tarfile
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from liver_raw import regions


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spatial-tar',type=Path,required=True);p.add_argument('--metadata-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    D=pd.read_csv(a.output/'liver_adapter_decomposition.csv');metadata=regions(a.metadata_dir);sections=sorted(metadata.section.unique());store={};names=None;tot=None;nspots=0
    with tarfile.open(a.spatial_tar) as tar:
        files={re.match(r'GSM\d+_(.+)_filtered_feature_bc_matrix\.h5',Path(n).name).group(1):n for n in tar.getnames() if n.endswith('filtered_feature_bc_matrix.h5')}
        for section in sections:
            with h5py.File(io.BytesIO(tar.extractfile(files[section]).read()),'r') as f:
                m=f['matrix'];M=sparse.csc_matrix((m['data'][:],m['indices'][:],m['indptr'][:]),shape=tuple(m['shape'][:]))
                bc=m['barcodes'].asstr()[:];ids=np.array([x.split('.')[0] for x in m['features/id'].asstr()[:]])
            if names is None:names=ids;tot=np.zeros(len(ids))
            elif not np.array_equal(names,ids):raise ValueError('spatial gene axis drift')
            C=M.T.tocsr().astype(np.float32);C=C.multiply((1e4/np.maximum(np.asarray(C.sum(1)).ravel(),1))[:,None]).tocsr();C.data=np.log1p(C.data)
            tot+=np.asarray(C.sum(0)).ravel();nspots+=C.shape[0]
            dd=metadata[metadata.section==section].set_index('barcode');common=dd.index.intersection(pd.Index(bc));index={b:i for i,b in enumerate(bc)};rows=np.array([index[b] for b in common]);dd=dd.loc[common]
            store[section]=(C[rows].tocsc(),(dd.region=='Scar').to_numpy(),(dd.region=='Hep').to_numpy())
    pool=np.flatnonzero(tot/nspots>.02);geneindex={g:i for i,g in enumerate(names)};positions={int(g):i for i,g in enumerate(pool)};moments={}
    for section,(matrix,_,_) in store.items():
        n=matrix.shape[0];sub=matrix[:,pool];mu=np.asarray(sub.sum(0)).ravel()/n;sq=sub.copy();sq.data=sq.data**2;sd=np.sqrt(np.maximum(np.asarray(sq.sum(0)).ravel()/n-mu**2,0));moments[section]=(mu,np.where(sd>0,sd,1.))
    def scores(genes):
        pc=np.array([positions[geneindex[g]] for g in genes if g in geneindex and geneindex[g] in positions]);out={}
        for section,(matrix,scar,hep) in store.items():
            if scar.sum()<10 or hep.sum()<10:continue
            mu,sd=moments[section];v=((np.asarray(matrix[:,pool[pc]].todense(),dtype=np.float32)-mu[pc])/sd[pc]).mean(1)
            out[section]=float(np.median(v[scar])-np.median(v[hep]))
        return pd.Series(out)
    rows=[]
    for threshold in [1.,1.25,1.5,2.]:
        core=scores(D.ensg[(D.pHSC>threshold)&(D.LX2>threshold)]);specific=scores(D.ensg[(D.pHSC>threshold)&(D.LX2.abs()<.25)])
        rows.append(pd.DataFrame(dict(section=core.index,core=core.values,spec=specific.reindex(core.index).values,threshold=threshold)))
    pd.concat(rows).round(6).to_csv(a.output/'liver_threshold_sensitivity_per_section.csv',index=False)
    print('threshold rows',sum(len(x) for x in rows),'genes in expression pool',len(pool),flush=True)
if __name__=='__main__':main()
