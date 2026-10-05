"""Historical numerical definitions, retained independently for regression verification."""

import numpy as np

def proj_simplex(v):
    u=np.sort(v)[::-1]; c=np.cumsum(u)-1; r=np.nonzero(u-c/np.arange(1,len(u)+1)>0)[0][-1]
    return np.maximum(v-c[r]/(r+1),0)

def fit_simplex(src,truth_k,ridge=.01,it=3000,tol=1e-12):
    n=src.shape[0]; uni=np.full(n,1/n)
    dsg=src.reshape(n,-1).T; out=truth_k.reshape(-1)
    G=dsg.T@dsg/len(out); l=dsg.T@out/len(out); L=2*(float(np.linalg.eigvalsh(G).max())+ridge); w=uni.copy()
    for _ in range(it):
        g=2*(G@w-l+ridge*(w-uni)); nw=proj_simplex(w-g/max(L,1e-12))
        if np.linalg.norm(nw-w)<=tol: return nw
        w=nw
    return w

def rowp(A,B):
    A=A-A.mean(-1,keepdims=True); B=B-B.mean(-1,keepdims=True)
    return (A*B).sum(-1)/(np.linalg.norm(A,axis=-1)*np.linalg.norm(B,axis=-1))

def sbasis(S,r=16):
    c=S-S.mean(0,keepdims=True); M=c.reshape(-1,c.shape[-1])
    _,s,vt=np.linalg.svd(M,full_matrices=False)
    nz=int((s>max(s[0]*1e-10,1e-12)).sum()); return vt[:min(r,nz)]

def fit_pair2(exp_,src,truth_k,cal,basis,ranks=(4,8,16)):
    bw=fit_simplex(exp_[:,cal],truth_k) if exp_.shape[0]>1 else np.ones(1)
    base=np.tensordot(bw,exp_,axes=(0,0)); res=truth_k-base[cal]; m=res.mean(0)
    gc=base+m
    sw=fit_simplex(src[:,cal],truth_k) if src.shape[0]>1 else np.ones(1)
    sc=np.tensordot(sw,src,axes=(0,0))
    cands=[(r,base+(m@basis[:min(r,len(basis))].T)@basis[:min(r,len(basis))]) for r in ranks]
    sc_=[(r,p,float(np.mean((p[cal]-truth_k)**2))) for r,p in cands]
    sel,prog,_=min(sc_,key=lambda x:x[2])
    bank=np.stack([base,gc,sc,prog]); cw=fit_simplex(bank[:,cal],truth_k)
    return base,np.tensordot(cw,bank,axes=(0,0)),cw,sel
