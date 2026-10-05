import numpy as np, pandas as pd
z = np.load("plate7_depth_sweep.npz", allow_pickle=True)
PB = z["PB"].astype(np.float32); names = [str(x) for x in z["names"]]; cellcount = z["cellcount"]
# layout verified earlier: C|line|HALF|LEVEL  and  T|line|drug ; levels index -> measured half-cells [10,10,20,40,80]
lines = sorted({n.split("|")[1] for n in names})
drugs = sorted({n.split("|")[2] for n in names if n.startswith("T|")})
idxT = {(n.split("|")[1], n.split("|")[2]): i for i, n in enumerate(names) if n.startswith("T|")}
idxC = {(n.split("|")[1], int(n.split("|")[3]), int(n.split("|")[2])): i
        for i, n in enumerate(names) if n.startswith("C|")}
nL, nD = len(lines), len(drugs)
WS = np.linspace(0, 2, 41)

def pcc(x, y):
    x = x - x.mean(-1, keepdims=True); y = y - y.mean(-1, keepdims=True)
    return (x * y).sum(-1) / np.sqrt((x * x).sum(-1) * (y * y).sum(-1) + 1e-30)

def sweep(bj, cj, tj):
    if bj.ndim == 1:
        return pcc(bj[None, :] + WS[:, None] * cj[None, :], tj[None, :])
    return pcc(bj[None] + WS[:, None, None] * cj[None, None, :], tj[None]).mean(-1)

rows = []
rng = np.random.default_rng(11)
for li in range(5):
    halfcells = int(np.median([cellcount[idxC[(L, li, h)]] for L in lines for h in (0, 1)]))
    T = np.stack([[PB[idxT[(L, d)]] for d in drugs] for L in lines]).astype(np.float32)
    CA = np.stack([PB[idxC[(L, li, 0)]] for L in lines]).astype(np.float32)
    CB = np.stack([PB[idxC[(L, li, 1)]] for L in lines]).astype(np.float32)
    d_build = T - CA[:, None, :]; d_truth = T - CB[:, None, :]
    S = d_build.sum(0); base = np.empty_like(d_build)
    for L in range(nL): base[L] = (S - d_build[L]) / (nL - 1)
    for s in range(20):
        for L in range(nL):
            perm = rng.permutation(nD); K, H = perm[:5], perm[5:]
            b = base[L][H]; t = d_truth[L][H]
            c = (d_build[L][K] - base[L][K]).mean(0)
            call = (d_build[L][H] - base[L][H]).mean(0)
            # honest LOO weight
            sc = np.zeros(len(WS))
            for j in range(5):
                idx = np.delete(np.arange(5), j)
                cj = (d_build[L][K[idx]] - base[L][K[idx]]).mean(0)
                sc += sweep(base[L][K[j]], cj, d_build[L][K[j]])
            wl = WS[int(np.argmax(sc))]
            rows.append(dict(level=li, halfcells=halfcells, total_control=2 * halfcells, split=s, line=L,
                             L0=float(pcc(b, t).mean()),
                             loo=float(pcc(b + wl * c, t).mean()), loo_w=float(wl),
                             fix03=float(pcc(b + 0.3 * c, t).mean()),
                             fix02=float(pcc(b + 0.2 * c, t).mean()),
                             oracle=float(sweep(b, c, t).max()),
                             ceil_fix03=float(pcc(b + 0.3 * call, t).mean()),
                             ceil_oracle=float(sweep(b, call, t).max())))
R = pd.DataFrame(rows); R.to_csv("depth_honest_raw.csv", index=False)
COLS = ["loo", "fix02", "fix03", "oracle", "ceil_fix03", "ceil_oracle"]
g = R.groupby(["level", "total_control"])[["L0"] + COLS + ["loo_w"]].mean()
for c in COLS: g["g_" + c] = g[c] - g.L0
out = g[["L0", "loo_w"] + ["g_" + c for c in COLS]].round(4)
out.to_csv("depth_honest_summary.csv")
print("GAIN over L0 at k=5, by control depth and weight rule (levels 0/1 are replicates at the same depth)")
print(out.to_string())
m = out.reset_index()
m = m[m.level != 0]                      # drop the duplicate rung so the span is a real 8x
print("\nspan over the real 4 rungs (20 -> 160 total control cells, 8x):")
for c in ("g_fix03", "g_ceil_fix03", "g_oracle", "g_ceil_oracle", "g_loo"):
    print("   %-14s %+.4f -> %+.4f   (x%.2f)" % (c, m[c].iloc[0], m[c].iloc[-1],
                                                 m[c].iloc[-1] / m[c].iloc[0] if m[c].iloc[0] != 0 else float("nan")))
