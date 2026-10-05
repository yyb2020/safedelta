"""Tahoe raw-derived pseudobulk -> atlas panel-level results.

Paper workflow caches the source-only basis per held-out context. The packaged
candidate implementation is checked against independent archived definitions.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from safedelta import pcc
from safedelta.adapter import source_basis,_candidate


def panels(A,B,draws,seed):
    rng=np.random.default_rng(seed);out=[];nC,nT,_=A.shape
    for i in range(nC):
        source=A[np.arange(nC)!=i];basis=source_basis(source)
        for repeat in range(draws):
            cal=rng.choice(nT,5,replace=False);ev=np.setdiff1d(np.arange(nT),cal)
            base,candidate,w,rank=_candidate(source,A[i,cal],cal,basis,'mixture');shared=[];independent=[]
            for j,t in enumerate(cal):
                keep=np.arange(5)!=j;b,c,_,_=_candidate(source,A[i,cal[keep]],cal[keep],basis,'mixture')
                for target,acc in [(A[i,t],shared),(B[i,t],independent)]:acc.append(pcc(c[t],target)-pcc(b[t],target))
            row=dict(ctx=i,evid_coupled=np.mean(shared),evid_coupled_sign=np.mean(np.array(shared)>0),evid_split=np.mean(independent),evid_split_sign=np.mean(np.array(independent)>0),true_gain=pcc(candidate[ev],B[i,ev]).mean()-pcc(base[ev],B[i,ev]).mean())
            out.append(row)
        print('criterion context',i+1,'/',nC,flush=True)
    return pd.DataFrame(out)


def ladder(A,B):
    nC,nT,_=A.shape;rng=np.random.default_rng(3);out=[]
    for i in range(nC):
        source=A[np.arange(nC)!=i];basis=source_basis(source);base=(A.sum(0)-A[i])/(nC-1)
        for repeat in range(12):
            cal=rng.choice(nT,5,replace=False);ev=np.setdiff1d(np.arange(nT),cal);y=A[i,cal]
            scalar=float((base[cal]*y).sum()/(base[cal]**2).sum())
            ab=np.linalg.lstsq(np.stack([base[cal].ravel(),np.ones(base[cal].size)]).T,y.ravel(),rcond=None)[0]
            residual=(y-base[cal]).mean(0);offset=base+residual;lowrank=base+(residual@basis.T)@basis
            b,c,w,_=_candidate(source,y,cal,basis,'mixture');ec=[];es=[]
            for j,t in enumerate(cal):
                keep=np.arange(5)!=j;lb,lc,_,_=_candidate(source,A[i,cal[keep]],cal[keep],basis,'mixture')
                ec.append(pcc(lc[t],A[i,t])-pcc(lb[t],A[i,t]));es.append(pcc(lc[t],B[i,t])-pcc(lb[t],B[i,t]))
            rc,rs=np.mean(ec)>0,np.mean(es)>0
            for mode,target in [('shared',A[i,ev]),('independent',B[i,ev])]:
                out.append(dict(ctx=i,mode=mode,L0_perturbation_mean=pcc(base[ev],target).mean(),L1_scalar=pcc(scalar*base[ev],target).mean(),L1i_scalar_intercept=pcc(ab[0]*base[ev]+ab[1],target).mean(),L2_global_offset=pcc(offset[ev],target).mean(),L2b_low_rank=pcc(lowrank[ev],target).mean(),L3_published_gate=pcc((c if rc else b)[ev],target).mean(),L3_repaired_gate=pcc((c if rs else b)[ev],target).mean(),w_global=w[1],w_lowrank=w[3],released_pub=rc,released_rep=rs))
        print('ladder context',i+1,'/',nC,flush=True)
    return pd.DataFrame(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--ladder',action='store_true');p.add_argument('--stage',choices=['all','plate2','plate7'],default='all');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    frames=[]
    if a.stage!='plate7':
        with np.load(a.data_dir/'plate2_pseudobulk.npz') as z:
            A=z['treated']-z['control_a'][:,None];B=z['treated']-z['control_b'][:,None]
            gf=panels(A,B,8,3);gf.to_csv(a.output/'safedelta_gate_evidence_repair.csv',index=False)
            frames.append(gf[['evid_split','true_gain']].assign(setting='Plate 2 (20 cells)'))
    else:
        gf=pd.read_csv(a.output/'safedelta_gate_evidence_repair.csv')
        frames.append(gf[['evid_split','true_gain']].assign(setting='Plate 2 (20 cells)'))
    if a.stage=='plate2':return
    with np.load(a.data_dir/'plate7_pseudobulk.npz') as z:
        A=z['deep_treated']-z['deep_control_a'][:,None];B=z['deep_treated']-z['deep_control_b'][:,None]
        deep=panels(A,B,10,7)[['ctx','evid_split','true_gain']].assign(setting='Plate 7 native (400 cells)')
        pd.concat([deep,*frames]).to_csv(a.output/'criterion_predicts_transfer.csv',index=False)
        if a.ladder:
            rows=[]
            for tag,name in [('deep','native 400 cells'),('shallow','downsampled 20 cells')]:
                A=z[tag+'_treated']-z[tag+'_control_a'][:,None];B=z[tag+'_treated']-z[tag+'_control_b'][:,None]
                rows.append(ladder(A,B).assign(depth=name))
            pd.concat(rows).to_csv(a.output/'plate7_depth_ladder.csv',index=False)
if __name__=='__main__':main()
