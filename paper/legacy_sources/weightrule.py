import numpy as np, pandas as pd
from scipy.stats import wilcoxon
z = np.load("plate7_16line_pseudobulk.npz", allow_pickle=True)
deep = z["deep"]; nL, nD = 16, 86
A = deep[:1376].reshape(nL, nD, -1).astype(np.float32)
C = deep[1376:1408].reshape(nL, 2, -1).astype(np.float32)
del z, deep
d_build = A - C[:, 0:1, :]; d_truth = A - C[:, 1:2, :]
del A, C
S = d_build.sum(0); base = np.empty_like(d_build)
for L in range(nL): base[L] = (S - d_build[L]) / (nL - 1)
WS = np.linspace(0, 2, 41)

def pcc(x, y):
    x = x - x.mean(-1, keepdims=True); y = y - y.mean(-1, keepdims=True)
    return (x * y).sum(-1) / np.sqrt((x * x).sum(-1) * (y * y).sum(-1) + 1e-30)

def sweep(bj, cj, tj):
    """mean PCC of bj + w*cj vs tj for every w in WS. bj/tj may be (G,) or (n,G)."""
    if bj.ndim == 1:
        P = bj[None, :] + WS[:, None] * cj[None, :]
        return pcc(P, tj[None, :])
    P = bj[None, :, :] + WS[:, None, None] * cj[None, None, :]
    return pcc(P, tj[None, :, :]).mean(-1)

FIXED = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
rows = []
rng = np.random.default_rng(7)
for k in (5, 20):
    for s in range(20):
        for L in range(nL):
            perm = rng.permutation(nD); K, H = perm[:k], perm[k:]
            b = base[L][H]; t = d_truth[L][H]
            c = (d_build[L][K] - base[L][K]).mean(0)
            call = (d_build[L][H] - base[L][H]).mean(0)
            rec = dict(k=k, split=s, line=L, L0=float(pcc(b, t).mean()))
            # (1) oracle weight: chosen on the held-out TRUTH   <- what my earlier scripts did
            rec["w_oracle"] = float(sweep(b, c, t).max())
            rec["w_oracle_arg"] = float(WS[int(np.argmax(sweep(b, c, t)))])
            # (2) LOO weight fitted inside the calibration set   <- honest
            sc = np.zeros(len(WS))
            for j in range(k):
                idx = np.delete(np.arange(k), j)
                cj = (d_build[L][K[idx]] - base[L][K[idx]]).mean(0)
                sc += sweep(base[L][K[j]], cj, d_build[L][K[j]])
            wl = WS[int(np.argmax(sc))]
            rec["w_loo"] = float(pcc(b + wl * c, t).mean()); rec["w_loo_arg"] = float(wl)
            # (3) fixed shrinkage, no fitting at all
            for w in FIXED: rec["fix%.1f" % w] = float(pcc(b + w * c, t).mean())
            # (4) ceiling under each weight rule (direction from ALL held-out drugs)
            rec["ceil_oracle"] = float(sweep(b, call, t).max())
            rec["ceil_fix0.3"] = float(pcc(b + 0.3 * call, t).mean())
            rows.append(rec)
R = pd.DataFrame(rows); R.to_csv("weightrule_raw.csv", index=False)
cols = ["w_oracle", "w_loo"] + ["fix%.1f" % w for w in FIXED] + ["ceil_oracle", "ceil_fix0.3"]
g = R.groupby("k")[["L0"] + cols + ["w_oracle_arg", "w_loo_arg"]].mean()
for c in cols: g["g_" + c] = g[c] - g.L0
print("GAIN over L0, by how the weight is chosen (independent controls, 320 panels per k)")
print(g[["L0"] + ["g_" + c for c in cols]].round(4).to_string())
print("\nmean chosen weight:  oracle %s   |  LOO %s"
      % (g.w_oracle_arg.round(2).to_dict(), g.w_loo_arg.round(2).to_dict()))
b5 = R[R.k == 5]
best_fixed = max(FIXED, key=lambda w: (b5["fix%.1f" % w] - b5.L0).mean())
print("\nbest FIXED weight at k=5: %.1f  ->  gain %+.4f" % (best_fixed, (b5["fix%.1f" % best_fixed] - b5.L0).mean()))
for nm in ("w_oracle", "w_loo"):
    d = b5["fix%.1f" % best_fixed] - b5[nm]
    print("   fixed-%.1f minus %-9s  %+.4f (95%% CI %+.4f..%+.4f)  positive %3d/%d"
          % (best_fixed, nm, d.mean(), d.mean() - 1.96 * d.sem(), d.mean() + 1.96 * d.sem(), (d > 0).sum(), len(d)))
g.round(5).to_csv("weightrule_summary.csv")
