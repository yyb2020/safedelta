import numpy as np, pandas as pd, sys
NPZ = sys.argv[1]
z = np.load(NPZ, allow_pickle=True)
deep = z["deep"]; nL, nD = 16, 86
A = deep[:1376].reshape(nL, nD, -1).astype(np.float32)
C = deep[1376:1408].reshape(nL, 2, -1).astype(np.float32)
del z, deep
d_build = A - C[:, 0:1, :]
d_truth = A - C[:, 1:2, :]
del A, C

def pcc(x, y):
    x = x - x.mean(-1, keepdims=True); y = y - y.mean(-1, keepdims=True)
    return (x * y).sum(-1) / np.sqrt((x * x).sum(-1) * (y * y).sum(-1) + 1e-30)

S = d_build.sum(0)
base = np.empty_like(d_build)
for L in range(nL):
    base[L] = (S - d_build[L]) / (nL - 1)
resid = d_truth - base

ctx = resid.mean(1, keepdims=True)
frac_ctx = float((np.broadcast_to(ctx, resid.shape) ** 2).sum() / (resid ** 2).sum())

WS = np.linspace(0, 2, 41)
def best_w(b, c, t):
    sc = [float(pcc(b + w * c, t).mean()) for w in WS]
    i = int(np.argmax(sc)); return WS[i], sc[i]

def lowrank(M, r):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]

rows = []
rng = np.random.default_rng(0)
for k in (2, 5, 10, 20, 40):
    for s in range(10):
        for L in range(nL):
            perm = rng.permutation(nD); K, H = perm[:k], perm[k:]
            b = base[L][H]; t = d_truth[L][H]
            p0 = float(pcc(b, t).mean())
            rk = d_build[L][K] - base[L][K]
            ck = rk.mean(0)
            # honest estimators
            sc_w1 = float(pcc(b + ck, t).mean())
            # ridge-style shrinkage: w fitted by leave-one-out inside the calibration set
            if k >= 3:
                loo = []
                for j in range(k):
                    cj = np.delete(rk, j, 0).mean(0)
                    bj = base[L][K[j]]; tj = d_build[L][K[j]]
                    loo.append([float(pcc(bj + w * cj, tj)) for w in WS])
                w_loo = WS[int(np.argmax(np.mean(loo, 0)))]
            else:
                w_loo = 1.0
            sc_loo = float(pcc(b + w_loo * ck, t).mean())
            # oracles
            w_ko, sc_ko = best_w(b, ck, t)
            _, sc_go = best_w(b, resid[L][H].mean(0), t)
            sc_full = float(pcc(b + resid[L][H], t).mean())
            # low-rank oracle on the held-out residual (rank 1..5)
            lr = {}
            for r in (1, 3, 5):
                lr[r] = float(pcc(b + lowrank(resid[L][H], r), t).mean())
            rows.append(dict(k=k, split=s, line=L, L0=p0, w1=sc_w1, loo=sc_loo, w_loo=w_loo,
                             oracle_w=sc_ko, w_oracle=w_ko, oracle_ctx=sc_go, oracle_full=sc_full,
                             oracle_lr1=lr[1], oracle_lr3=lr[3], oracle_lr5=lr[5]))
R = pd.DataFrame(rows)
R.to_csv("headroom_raw.csv", index=False)
g = R.groupby("k").mean(numeric_only=True)
g["gain_w1"] = g.w1 - g.L0; g["gain_loo"] = g.loo - g.L0
g["gain_oracle_w"] = g.oracle_w - g.L0; g["gain_oracle_ctx"] = g.oracle_ctx - g.L0
g["gain_oracle_lr1"] = g.oracle_lr1 - g.L0; g["gain_oracle_lr3"] = g.oracle_lr3 - g.L0
g["gain_oracle_full"] = g.oracle_full - g.L0
g.round(5).to_csv("headroom_by_k.csv")
with open("headroom_meta.txt", "w") as f:
    f.write("frac_ctx=%.4f\nL0=%.4f\nn_rows=%d\n" % (frac_ctx, R.L0.mean(), len(R)))
print("frac of residual variance that is context-level: %.3f" % frac_ctx)
print(g[["L0", "gain_w1", "gain_loo", "gain_oracle_w", "gain_oracle_ctx",
         "gain_oracle_lr1", "gain_oracle_lr3", "gain_oracle_full"]].round(4).to_string())
