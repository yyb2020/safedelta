"""Summarise JOB-W10-A outputs: acceptance checks 1-4, per-method inflation, ranking change,
dot-gap vs a_proj regression.  Writes out/w10_summary.md and out/w10_rank_table.csv."""
import sys, os, glob
import numpy as np, pandas as pd
from scipy import stats

OUT = "/projects/xunixibao/handoff/out"
M_ALL = ["trVAE", "trVAE_plus_offset", "trVAE_plus_calmean", "trainMean_theirs",
         "pooled_B", "calibration_mean", "B_plus_offset"]
lines = []
P = lambda s="": lines.append(s)

rank_rows = []
for f in sorted(glob.glob(os.path.join(OUT, "w10_trvae_*.csv")) + glob.glob(os.path.join(OUT, "w10_scpram_*.csv")) + glob.glob(os.path.join(OUT, "w10_scgen_*.csv"))):
    if f.endswith(("_sensitivity.csv", "_meta.csv")):
        continue
    base = os.path.basename(f)[:-4]
    name = base.split("_", 2)[2] + " [" + base.split("_")[1] + "]"
    R = pd.read_csv(f)
    DM = "trVAE" if "trVAE__pcc" in R.columns else ("scPRAM" if "scPRAM__pcc" in R.columns else "scGen")
    if DM + "__pcc" not in R.columns or "pooled_B__pcc" not in R.columns or not os.path.exists(f.replace(".csv", "_sensitivity.csv")):
        print("skip (incomplete):", base); continue
    R = R.rename(columns={c: c.replace(DM + "_", "trVAE_").replace(DM + "__", "trVAE__") for c in R.columns})
    S_tmp = None
    S = pd.read_csv(f.replace(".csv", "_sensitivity.csv"))
    S["method"] = S["method"].str.replace("^" + DM, "trVAE", regex=True)
    Mt = pd.read_csv(f.replace(".csv", "_meta.csv"))
    M = [m for m in M_ALL if m + "__pcc" in R.columns]
    P(f"\n## {name}: model column shown as trVAE = {DM}; contexts={R.context.nunique()} perturbations={R.target.nunique()} "
      f"splits={R.split.nunique()} folds={R.fold.nunique()} n0 median={int(R.n0.median())} "
      f"epochs run (median)={int(R.epochs_run.median())}")
    # ---- check 1: a=0 estimator identical after fold-averaging
    g = R.groupby(["context", "target", "split", "regime"])[[m + "__pcc" for m in M] + [m + "__dot" for m in M]].mean()
    sh, ind = g.xs("shared", level="regime"), g.xs("independent", level="regime")
    d1p = (sh["pooled_B__pcc"] - ind["pooled_B__pcc"]).abs().max()
    d1d = (sh["pooled_B__dot"] - ind["pooled_B__dot"]).abs().max()
    P(f"- check1 (pooled_B fold-averaged, shared vs independent): max|dPCC|={d1p:.3e}, max|ddot|={d1d:.3e}")
    # ---- check 2: prediction identical across regimes — by construction (same array scored twice);
    # evidence: the stored predictions npz has ONE array per (context, split, fold, target).
    npz = np.load(f.replace(".csv", "_predictions.npz"))
    P(f"- check2: predictions stored once per (context,split,fold,target): {sum(k.startswith('P|') for k in npz.files)} arrays; "
      f"both regimes are scored from the same array (see run_w10_trvae.py).")
    # ---- check 3: fold-averaged dot gap == a_proj_centred * ||cA-cB||^2_centred / 2 (exact identity)
    gap = sh[[m + "__dot" for m in M]] - ind[[m + "__dot" for m in M]]
    S2 = S.set_index(["context", "target", "split"])
    for m in M:
        sub = S2[S2.method == m]
        pred = sub["a_proj_centred"] * sub["dc_centred"] ** 2 / 2
        ratio = (gap[m + "__dot"] / pred.reindex(gap.index)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio):
            P(f"- check3 ratio dot_gap/(a_proj·‖Δc‖²/2) {m:20s}: mean={ratio.mean():.5f} sd={ratio.std():.2e} n={len(ratio)}")
        else:
            P(f"- check3 {m}: undefined (a=0 → 0/0)")
    # ---- check 4: a_hat variance
    P("- check4 sensitivity probe (per method: mean ± sd over contexts×splits×targets):")
    P("  | method | a_hat (norm ratio) | a_proj (projection) | cos |")
    P("  |---|---|---|---|")
    for m in M:
        s = S[S.method == m]
        P(f"  | {m} | {s.a_hat.mean():.3f} ± {s.a_hat.std():.3f} | {s.a_proj_centred.mean():.3f} ± {s.a_proj_centred.std():.3f} | {s['cos'].mean():.3f} |")
    # ---- inflation per method on TEST perturbations (fold-averaged, then mean over panels)
    Rt = R[R.in_test]
    gt = Rt.groupby(["context", "target", "split", "regime"])[[m + "__pcc" for m in M] + [m + "__r2" for m in M] + [m + "__dot" for m in M]].mean()
    sht, indt = gt.xs("shared", level="regime"), gt.xs("independent", level="regime")
    P(f"- test panels (context×target×split, fold-averaged): {len(sht)}")
    P("  | method | PCC shared | PCC independent | ΔPCC | rel. inflation | R² shared | R² indep | dot gap (mean) | rank shared → indep |")
    P("  |---|---|---|---|---|---|---|---|---|")
    ps = sht[[m + "__pcc" for m in M]].mean(); pi = indt[[m + "__pcc" for m in M]].mean()
    rs = ps.rank(ascending=False); ri = pi.rank(ascending=False)
    for m in M:
        a, b = ps[m + "__pcc"], pi[m + "__pcc"]
        P(f"  | {m} | {a:.4f} | {b:.4f} | {a-b:+.4f} | {(a-b)/abs(b)*100:+.1f}% | {sht[m+'__r2'].mean():.4f} | {indt[m+'__r2'].mean():.4f} | "
          f"{(sht[m+'__dot']-indt[m+'__dot']).mean():+.3f} | {int(rs[m+'__pcc'])} → {int(ri[m+'__pcc'])} |")
        rank_rows.append(dict(dataset=name, method=m, pcc_shared=a, pcc_independent=b, dpcc=a - b,
                              rel_inflation=(a - b) / abs(b), r2_shared=sht[m + '__r2'].mean(), r2_independent=indt[m + '__r2'].mean(),
                              dot_gap=(sht[m + '__dot'] - indt[m + '__dot']).mean(),
                              rank_shared=int(rs[m + '__pcc']), rank_independent=int(ri[m + '__pcc']), n_panels=len(sht)))
    # ---- zero-shot layer (a=1 layer of the task sheet): ALL perturbations, methods needing no calibration
    Z = ["trVAE", "trainMean_theirs", "pooled_B"]
    gz = R.groupby(["context", "target", "split", "regime"])[[m + "__pcc" for m in Z] + [m + "__dot" for m in Z]].mean()
    shz, inz = gz.xs("shared", level="regime"), gz.xs("independent", level="regime")
    P(f"- zero-shot layer, ALL perturbations (panels={len(shz)}, contexts={R.context.nunique()}):")
    P("  | method | PCC shared | PCC independent | ΔPCC | rel. | dot gap | rank shared → indep |")
    P("  |---|---|---|---|---|---|---|")
    pz = shz[[m + "__pcc" for m in Z]].mean(); qz = inz[[m + "__pcc" for m in Z]].mean()
    rz, rq = pz.rank(ascending=False), qz.rank(ascending=False)
    for m in Z:
        a, b = pz[m + "__pcc"], qz[m + "__pcc"]
        P(f"  | {m} | {a:.4f} | {b:.4f} | {a-b:+.4f} | {(a-b)/abs(b)*100:+.1f}% | {(shz[m+'__dot']-inz[m+'__dot']).mean():+.3f} | {int(rz[m+'__pcc'])} → {int(rq[m+'__pcc'])} |")
    pcz = (shz - inz).groupby(level="context").mean()
    dvz, dtz = pcz["trVAE__pcc"], pcz["trainMean_theirs__pcc"]
    P(f"  per-context ΔPCC(trVAE) > ΔPCC(trainMean_theirs) in {(dvz > dtz).sum()}/{len(dvz)} contexts; "
      f"per-context ΔPCC(trVAE) > 0 in {(dvz > 0).sum()}/{len(dvz)}")
    # ---- paired test: trVAE inflation vs pooled_B inflation per context
    pc = (sht - indt).groupby(level="context").mean()
    dv, db = pc["trVAE__pcc"], pc["pooled_B__pcc"]
    if len(dv) >= 3:
        try:
            w = stats.wilcoxon(dv - db)
            P(f"- per-context ΔPCC(trVAE) > ΔPCC(pooled_B) in {(dv > db).sum()}/{len(dv)} contexts; Wilcoxon P={w.pvalue:.3g}")
        except Exception as e:
            P(f"- per-context comparison: {e}")
    # ---- dot-gap vs a_proj regression across (context, split, method, target) — slope should be 1
    recs = []
    for m in M:
        sub = S2[S2.method == m]
        x = (sub["a_proj_centred"] * sub["dc_centred"] ** 2 / 2).reindex(gap.index)
        y = gap[m + "__dot"]
        ok = x.notna() & y.notna()
        recs.append(pd.DataFrame({"x": x[ok], "y": y[ok], "method": m}))
    D = pd.concat(recs)
    D = D[D.x.abs() > 0]
    if len(D) > 3:
        sl = stats.linregress(D.x, D.y)
        P(f"- regression fold-averaged dot gap ~ a_proj·‖Δc‖²/2 over all methods: slope={sl.slope:.5f} intercept={sl.intercept:.2e} R²={sl.rvalue**2:.5f}")
    # ---- lambda
    P(f"- λ (noise_centred/signal) median over contexts×splits: {Mt.lam.median():.4f}; n0 median {int(Mt.n0.median())}")

pd.DataFrame(rank_rows).to_csv(os.path.join(OUT, "w10_rank_table.csv"), index=False)
hdr = ["# W10 summary (auto-generated by _w10/analyze_w10.py)",
       "PCC = centred Pearson of pseudobulk deltas over the benchmark's 5000 HVGs (their `pearson_distance` on DEG=5000).",
       "Equal marginal depth: prediction-side and truth-side controls are both n/2 plates of the held-out context; cross-fitted over 2 folds × splits."]
open(os.path.join(OUT, "w10_summary.md"), "w").write("\n".join(hdr + lines) + "\n")
print("\n".join(hdr + lines))
