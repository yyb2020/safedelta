"""B2: compare our reproduction of the benchmark's published methods (zero-shot layer, all
perturbations) with the published per-op PCC (Results/Cellular_context_ood/
cellular_ood_performance_top5000.csv, DEG=5000), paired by op = <perturbation>_<context>,
and show where the head of the published leaderboard lands under independent controls.
Writes out/w10_published_paired.csv and out/w10_published_leaderboard.csv."""
import os, glob
import numpy as np, pandas as pd

OUT = "/projects/xunixibao/handoff/out"
PUB = pd.read_csv("/projects/xunixibao/handoff/_w10/scPerturBench/Results/Cellular_context_ood/cellular_ood_performance_top5000.csv")
PUB["cor"] = PUB["cor"].astype(float)

paired, board = [], []
ours = {}   # (dataset, method) -> per-op DataFrame with shared/independent
for f in sorted(glob.glob(os.path.join(OUT, "w10_*_*.csv"))):
    b = os.path.basename(f)
    if b.endswith(("_sensitivity.csv", "_meta.csv")) or "_iid" in b or "cellsplit" in b or b.startswith(("w10_rank", "w10_published")):
        continue
    parts = b[:-4].split("_", 2)
    if len(parts) < 3:
        continue
    driver, ds = parts[1], parts[2]
    R = pd.read_csv(f)
    model = "trVAE" if "trVAE__pcc" in R.columns else ("scPRAM" if "scPRAM__pcc" in R.columns else "scGen")
    if model + "__pcc" not in R.columns or "pooled_B__pcc" not in R.columns:
        print("skip (incomplete columns):", b); continue
    cols = {model: model + "__pcc", "trainMean": "trainMean_theirs__pcc", "pooled_B": "pooled_B__pcc"}
    g = R.groupby(["context", "target", "regime"])[list(cols.values())].mean().reset_index()
    g["op"] = g["target"] + "_" + g["context"]
    for m, col in cols.items():
        if m == "trainMean" and driver != "trvae":
            continue          # trainMean is identical across drivers; keep the trvae copy
        w = g.pivot(index="op", columns="regime", values=col)
        ours[(ds, m)] = w
        pub = PUB[(PUB.DataSet == ds) & (PUB.method == m)].set_index("op")["cor"]
        j = w.join(pub, how="inner")
        if m != "pooled_B" and len(j) >= 3:
            d = j["shared"] - j["cor"]
            paired.append(dict(dataset=ds, method=m, n_ops_ours=len(w), n_ops_published=len(pub), n_paired=len(j),
                               ours_shared_mean=j["shared"].mean(), published_mean_on_paired=j["cor"].mean(),
                               ours_independent_mean=j["independent"].mean(),
                               paired_diff_median=d.median(), paired_diff_q25=d.quantile(.25), paired_diff_q75=d.quantile(.75),
                               pearson_shared_vs_published=np.corrcoef(j["shared"], j["cor"])[0, 1],
                               inflation_rel=(j["shared"].mean() - j["independent"].mean()) / abs(j["independent"].mean())))

P = pd.DataFrame(paired)
P.to_csv(os.path.join(OUT, "w10_published_paired.csv"), index=False)

# leaderboard: published means (all 14) + our shared/independent for the methods we ran, on our op set
for ds in sorted(set(k[0] for k in ours)):
    pubm = PUB[PUB.DataSet == ds].groupby("method")["cor"].mean().sort_values(ascending=False)
    ops_ours = None
    for (d2, m), w in ours.items():
        if d2 == ds:
            ops_ours = w.index if ops_ours is None else ops_ours.intersection(w.index)
    for rank, (m, c) in enumerate(pubm.items(), 1):
        row = dict(dataset=ds, method=m, published_rank=rank, published_mean_cor=c, published_mean_on_our_ops=np.nan,
                   ours_shared=np.nan, ours_independent=np.nan, n_ops=len(ops_ours) if ops_ours is not None else 0)
        pub_ops = PUB[(PUB.DataSet == ds) & (PUB.method == m)].set_index("op")["cor"]
        if ops_ours is not None:
            row["published_mean_on_our_ops"] = pub_ops.reindex(ops_ours).mean()
        if (ds, m) in ours:
            w = ours[(ds, m)].reindex(ops_ours)
            row["ours_shared"] = w["shared"].mean(); row["ours_independent"] = w["independent"].mean()
        board.append(row)
    if (ds, "pooled_B") in ours:
        w = ours[(ds, "pooled_B")].reindex(ops_ours)
        board.append(dict(dataset=ds, method="pooled_B (delta-space, a=0; NOT in benchmark)", published_rank=np.nan,
                          published_mean_cor=np.nan, published_mean_on_our_ops=np.nan,
                          ours_shared=w["shared"].mean(), ours_independent=w["independent"].mean(), n_ops=len(ops_ours)))
B = pd.DataFrame(board)
B.to_csv(os.path.join(OUT, "w10_published_leaderboard.csv"), index=False)

pd.set_option("display.width", 200)
print("=== paired reproduction (zero-shot layer, shared regime vs published, by op)")
print(P.round(4).to_string(index=False))
print("\n=== leaderboards (our numbers on the common op set)")
for ds in B.dataset.unique():
    print(f"\n-- {ds}")
    print(B[B.dataset == ds][["method", "published_rank", "published_mean_cor", "published_mean_on_our_ops", "ours_shared", "ours_independent"]].round(4).to_string(index=False))
