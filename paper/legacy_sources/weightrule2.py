"""Head-to-head on the SAME data and the SAME panels: only the weight rule differs.

  A) the delivered pipeline's rule  -- simplex least squares over a 4-candidate library
     with a ridge pull toward UNIFORM, solved by the capsule's own projected-gradient
     routine (project_simplex / fit_simplex, transcribed from
     code/scripts/run_safedelta_v3_nested_utility_development_v1.py)
  B) LOO-fitted scalar weight on the global offset
  C) fixed scalar shrinkage, no fitting at all
  D) oracle scalar weight (reference only -- peeks at held-out truth)

NOTE: this runs on the 16-line reconstruction, NOT on the delivered 15-context ladder,
whose raw vectors are not in hand. Conclusions transfer only as far as that.
"""
import numpy as np, pandas as pd, sys
from scipy.stats import wilcoxon
z = np.load("plate7_16line_pseudobulk.npz", allow_pickle=True)
deep = z["deep"]; nL, nD = 16, 86
A = deep[:1376].reshape(nL, nD, -1).astype(np.float32)
C = deep[1376:1408].reshape(nL, 2, -1).astype(np.float32)
del z, deep
d_build = A - C[:, 0:1, :]; d_truth = A - C[:, 1:2, :]
ctrl = (C[:, 0, :] + C[:, 1, :]) / 2.0
del A, C
S = d_build.sum(0); base = np.empty_like(d_build)
for L in range(nL): base[L] = (S - d_build[L]) / (nL - 1)
G = base.shape[2]
WS = np.linspace(0, 2, 41)

def pcc(x, y):
    x = x - x.mean(-1, keepdims=True); y = y - y.mean(-1, keepdims=True)
    return (x * y).sum(-1) / np.sqrt((x * x).sum(-1) * (y * y).sum(-1) + 1e-30)

def sweep(bj, cj, tj):
    if bj.ndim == 1:
        return pcc(bj[None, :] + WS[:, None] * cj[None, :], tj[None, :])
    return pcc(bj[None] + WS[:, None, None] * cj[None, None, :], tj[None]).mean(-1)

# ---- the capsule's solver, transcribed verbatim in structure ----
def project_simplex(values):
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    valid = ordered - (cumulative - 1.0) / np.arange(1, len(values) + 1) > 0
    rho = np.flatnonzero(valid)[-1]
    return np.maximum(values - (cumulative[rho] - 1.0) / (rho + 1), 0.0)

def fit_simplex(design, outcome, ridge, iters=400, tol=1e-9):
    """design (m, n_obs), outcome (n_obs,) -> simplex weights pulled toward uniform"""
    count = design.shape[0]
    uniform = np.full(count, 1.0 / count)
    gram = design @ design.T / len(outcome)
    linear = design @ outcome / len(outcome)
    lipschitz = 2.0 * (float(np.linalg.eigvalsh(gram).max()) + ridge)
    weight = uniform.copy()
    for _ in range(iters):
        updated = project_simplex(weight - 2.0 * (gram @ weight - linear + ridge * (weight - uniform)) / lipschitz)
        if np.linalg.norm(updated - weight) <= tol:
            return updated
        weight = updated
    return weight

rows = []
rng = np.random.default_rng(303)
RIDGES = [1e-3, 1e-2, 1e-1]
FIXED = [0.1, 0.2, 0.3, 0.4]
for k in (5, 20):
    for s in range(20):
        for L in range(nL):
            perm = rng.permutation(nD); K, H = perm[:k], perm[k:]
            b = base[L][H]; t = d_truth[L][H]
            rec = dict(k=k, split=s, line=L, L0=float(pcc(b, t).mean()))
            Rk = d_build[L][K] - base[L][K]                 # calibration residual (build-side)

            # --- the 4-candidate library, exactly as the delivered pipeline defines it ---
            cand_H = [np.zeros(G)]                          # 1. baseline (zero correction)
            cand_K = [np.zeros((k, G))]
            off = Rk.mean(0)                                # 2. global offset
            cand_H.append(off); cand_K.append(np.tile(off, (k, 1)))
            # 3. source: nearest other context by control-profile distance, its residual
            dist = np.linalg.norm(ctrl - ctrl[L], axis=1); dist[L] = np.inf
            src = int(np.argmin(dist))
            src_corr = d_build[src] - base[src]
            cand_H.append(src_corr[H].mean(0)); cand_K.append(src_corr[K])
            # 4. low-rank: rank-1 of the calibration residual
            U, sv, Vt = np.linalg.svd(Rk, full_matrices=False)
            lr = (U[:, :1] * sv[:1]) @ Vt[:1]
            cand_H.append(lr.mean(0)); cand_K.append(lr)

            design = np.stack([c.reshape(-1) if c.ndim == 2 else np.tile(c, (k, 1)).reshape(-1)
                               for c in cand_K])            # (4, k*G)
            outcome = Rk.reshape(-1)
            for rg in RIDGES:
                w = fit_simplex(design, outcome, rg)
                corr = sum(w[i] * cand_H[i] for i in range(4))
                rec["simplex_r%g" % rg] = float(pcc(b + corr, t).mean())
                rec["w_off_r%g" % rg] = float(w[1]); rec["w_base_r%g" % rg] = float(w[0])

            # --- B) LOO scalar on the global offset ---
            sc = np.zeros(len(WS))
            for j in range(k):
                idx = np.delete(np.arange(k), j)
                cj = (d_build[L][K[idx]] - base[L][K[idx]]).mean(0)
                sc += sweep(base[L][K[j]], cj, d_build[L][K[j]])
            rec["loo"] = float(pcc(b + WS[int(np.argmax(sc))] * off, t).mean())

            # --- C) fixed shrinkage, no fitting ---
            for w_ in FIXED: rec["fix%.1f" % w_] = float(pcc(b + w_ * off, t).mean())
            # --- D) oracle reference ---
            rec["oracle"] = float(sweep(b, off, t).max())
            rows.append(rec)

R = pd.DataFrame(rows); R.to_csv("weightrule2_raw.csv", index=False)
METH = ["simplex_r0.001", "simplex_r0.01", "simplex_r0.1", "loo"] + ["fix%.1f" % w for w in FIXED] + ["oracle"]
g = R.groupby("k")[["L0"] + METH].mean()
for c in METH: g["g_" + c] = g[c] - g.L0
print("GAIN over L0 -- same data, same panels, ONLY the weight rule differs")
print(g[["L0"] + ["g_" + c for c in METH]].round(4).to_string())
print("\nmean weight the delivered rule puts on the global offset (ridge 0.01): %s"
      % R.groupby("k")["w_off_r0.01"].mean().round(3).to_dict())
print("mean weight it leaves on the baseline            (ridge 0.01): %s"
      % R.groupby("k")["w_base_r0.01"].mean().round(3).to_dict())
for k in (5, 20):
    gk = R[R.k == k]
    best_fix = max(FIXED, key=lambda w: (gk["fix%.1f" % w] - gk.L0).mean())
    d = gk["fix%.1f" % best_fix] - gk["simplex_r0.01"]
    print("\nk=%d  best fixed w=%.1f  minus  delivered simplex rule (ridge 0.01):" % (k, best_fix))
    print("   delta %+.4f (95%% CI %+.4f..%+.4f)  positive %3d/%d  Wilcoxon P=%.2e"
          % (d.mean(), d.mean() - 1.96 * d.sem(), d.mean() + 1.96 * d.sem(), (d > 0).sum(), len(d), wilcoxon(d)[1]))
g.round(5).to_csv("weightrule2_summary.csv")
