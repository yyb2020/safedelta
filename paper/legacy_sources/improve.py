import numpy as np, pandas as pd, sys
z = np.load(sys.argv[1], allow_pickle=True)
deep = z["deep"]; genes = np.asarray(z["genes"]); nL, nD = 16, 86
A = deep[:1376].reshape(nL, nD, -1).astype(np.float32)
C = deep[1376:1408].reshape(nL, 2, -1).astype(np.float32)
del z, deep
d_build = A - C[:, 0:1, :]
d_truth = A - C[:, 1:2, :]
c_diff = (C[:, 0, :] - C[:, 1, :]) / 2.0          # observable proxy for the control error
del A, C

def pcc(x, y):
    x = x - x.mean(-1, keepdims=True); y = y - y.mean(-1, keepdims=True)
    return (x * y).sum(-1) / np.sqrt((x * x).sum(-1) * (y * y).sum(-1) + 1e-30)

S = d_build.sum(0)
base = np.empty_like(d_build)
for L in range(nL):
    base[L] = (S - d_build[L]) / (nL - 1)
resid = d_truth - base

# context offsets of the OTHER contexts, estimable without the target's control
off_all = resid.mean(1)                                    # (nL, G) -- oracle offsets, used only for the subspace
WS = np.linspace(0, 2, 41)

def score(b, c, t, w):
    return float(pcc(b + w * c, t).mean())

def fit_w_loo(L, K, rk, ck_fn):
    k = len(K)
    if k < 3: return 1.0
    m = []
    for j in range(k):
        cj = ck_fn(np.delete(np.arange(k), j))
        bj = base[L][K[j]]; tj = d_build[L][K[j]]
        m.append([float(pcc(bj + w * cj, tj)) for w in WS])
    return WS[int(np.argmax(np.mean(m, 0)))]

rows = []
rng = np.random.default_rng(1)
for k in (5, 10, 20):
    for s in range(10):
        for L in range(nL):
            perm = rng.permutation(nD); K, H = perm[:k], perm[k:]
            b = base[L][H]; t = d_truth[L][H]
            p0 = float(pcc(b, t).mean())
            rk = d_build[L][K] - base[L][K]
            raw = rk.mean(0)

            # --- A. per-gene reliability shrinkage using the observable control-half difference
            v_ctrl = float((c_diff[L] ** 2).mean())
            g_ctrl = c_diff[L] ** 2
            v_sig = np.maximum(raw ** 2 - g_ctrl, 0.0)
            shrink = v_sig / (v_sig + g_ctrl + 1e-8)
            corr_shr = raw * shrink

            # --- B. projection onto the subspace spanned by the OTHER contexts' offsets
            others = np.delete(np.arange(nL), L)
            M = off_all[others]                                   # (15, G)
            M = M - M.mean(0, keepdims=True)
            U, sv, Vt = np.linalg.svd(M, full_matrices=False)
            projs = {}
            for r in (1, 2, 3, 5):
                Vr = Vt[:r]
                projs[r] = (raw @ Vr.T) @ Vr

            # --- C. both
            both = {}
            for r in (1, 3):
                Vr = Vt[:r]
                both[r] = (corr_shr @ Vr.T) @ Vr

            rec = dict(k=k, split=s, line=L, L0=p0)
            for nm, c in [("raw", raw), ("shrink", corr_shr)] + \
                         [("proj%d" % r, projs[r]) for r in (1, 2, 3, 5)] + \
                         [("both%d" % r, both[r]) for r in (1, 3)]:
                w = fit_w_loo(L, K, rk, lambda idx, c=c: c)      # weight fitted honestly on calibration
                rec[nm] = score(b, c, t, w); rec["w_" + nm] = w
            rec["oracle_ctx"] = max(score(b, resid[L][H].mean(0), t, w) for w in WS)
            rows.append(rec)
R = pd.DataFrame(rows)
R.to_csv("improve_raw.csv", index=False)
g = R.groupby("k").mean(numeric_only=True)
cols = ["raw", "shrink", "proj1", "proj2", "proj3", "proj5", "both1", "both3", "oracle_ctx"]
for c in cols: g["gain_" + c] = g[c] - g.L0
print(g[["L0"] + ["gain_" + c for c in cols]].round(4).to_string())
g.round(5).to_csv("improve_by_k.csv")

# what is the rank-1 axis biologically?
M = off_all - off_all.mean(0, keepdims=True)
U, sv, Vt = np.linalg.svd(M, full_matrices=False)
ev = sv ** 2 / (sv ** 2).sum()
print("\ncontext-offset spectrum (fraction of variance): %s" % np.round(ev[:6], 3))
ax = Vt[0]
o = np.argsort(ax)
pd.DataFrame({"gene": np.r_[genes[o[:40]], genes[o[-40:]]],
              "loading": np.r_[ax[o[:40]], ax[o[-40:]]],
              "end": ["negative"] * 40 + ["positive"] * 40}).to_csv("offset_axis_genes.csv", index=False)
print("top +: %s" % ", ".join(map(str, genes[o[-15:]][::-1])))
print("top -: %s" % ", ".join(map(str, genes[o[:15]])))
