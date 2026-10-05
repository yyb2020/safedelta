"""Does the delivered rule lose because of its WEIGHTS or because of its CANDIDATE LIBRARY?
Same solver (capsule fit_simplex, ridge->uniform), same data, same panels; only the library varies."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
z=np.load("plate7_16line_pseudobulk.npz",allow_pickle=True); deep=z["deep"]; nL,nD=16,86
A=deep[:1376].reshape(nL,nD,-1).astype(np.float32); C=deep[1376:1408].reshape(nL,2,-1).astype(np.float32)
del z,deep
d_build=A-C[:,0:1,:]; d_truth=A-C[:,1:2,:]; ctrl=(C[:,0,:]+C[:,1,:])/2.0; del A,C
S=d_build.sum(0); base=np.empty_like(d_build)
for L in range(nL): base[L]=(S-d_build[L])/(nL-1)
G=base.shape[2]
def pcc(x,y):
    x=x-x.mean(-1,keepdims=True); y=y-y.mean(-1,keepdims=True)
    return (x*y).sum(-1)/np.sqrt((x*x).sum(-1)*(y*y).sum(-1)+1e-30)
def proj(v):
    o=np.sort(v)[::-1]; cs=np.cumsum(o)
    ok=o-(cs-1.0)/np.arange(1,len(v)+1)>0; r=np.flatnonzero(ok)[-1]
    return np.maximum(v-(cs[r]-1.0)/(r+1),0.0)
def fit_simplex(D,y,rg,iters=400,tol=1e-9):
    m=D.shape[0]; u=np.full(m,1.0/m); Gm=D@D.T/len(y); lin=D@y/len(y)
    lip=2.0*(float(np.linalg.eigvalsh(Gm).max())+rg); w=u.copy()
    for _ in range(iters):
        nw=proj(w-2.0*(Gm@w-lin+rg*(w-u))/lip)
        if np.linalg.norm(nw-w)<=tol: return nw
        w=nw
    return w
rows=[]; rng=np.random.default_rng(404)
for k in (5,20):
    for s in range(20):
        for L in range(nL):
            perm=rng.permutation(nD); K,H=perm[:k],perm[k:]
            b=base[L][H]; t=d_truth[L][H]; Rk=d_build[L][K]-base[L][K]
            off=Rk.mean(0)
            dist=np.linalg.norm(ctrl-ctrl[L],axis=1); dist[L]=np.inf; src=int(np.argmin(dist))
            sc_=d_build[src]-base[src]
            U,sv,Vt=np.linalg.svd(Rk,full_matrices=False); lr=(U[:,:1]*sv[:1])@Vt[:1]
            libs={"L2_zero_offset":[(np.zeros((k,G)),np.zeros(G)),(np.tile(off,(k,1)),off)],
                  "L3_plus_source":[(np.zeros((k,G)),np.zeros(G)),(np.tile(off,(k,1)),off),(sc_[K],sc_[H].mean(0))],
                  "L4_full":[(np.zeros((k,G)),np.zeros(G)),(np.tile(off,(k,1)),off),(sc_[K],sc_[H].mean(0)),(lr,lr.mean(0))]}
            rec=dict(k=k,split=s,line=L,L0=float(pcc(b,t).mean()),fix03=float(pcc(b+0.3*off,t).mean()))
            for nm,lib in libs.items():
                D=np.stack([c[0].reshape(-1) for c in lib]); w=fit_simplex(D,Rk.reshape(-1),1e-2)
                corr=sum(w[i]*lib[i][1] for i in range(len(lib)))
                rec[nm]=float(pcc(b+corr,t).mean()); rec["woff_"+nm]=float(w[1])
            rows.append(rec)
R=pd.DataFrame(rows); R.to_csv("lib_ablate_raw.csv",index=False)
M=["L2_zero_offset","L3_plus_source","L4_full","fix03"]
g=R.groupby("k")[["L0"]+M].mean()
for c in M: g["g_"+c]=g[c]-g.L0
print("SAME solver, SAME panels; only the candidate library changes")
print(g[["L0"]+["g_"+c for c in M]].round(4).to_string())
print("\nweight the solver puts on the offset, by library (k=5): %s"
      %{c:round(R[R.k==5]["woff_"+c].mean(),3) for c in M[:3]})
g5=R[R.k==5]
for c in ("L2_zero_offset","L3_plus_source","L4_full"):
    d=g5.fix03-g5[c]
    print("   fixed-0.3 minus %-16s %+.4f (95%% CI %+.4f..%+.4f) positive %3d/%d P=%.1e"
          %(c,d.mean(),d.mean()-1.96*d.sem(),d.mean()+1.96*d.sem(),(d>0).sum(),len(d),wilcoxon(d)[1]))
g.round(5).to_csv("lib_ablate_summary.csv")
