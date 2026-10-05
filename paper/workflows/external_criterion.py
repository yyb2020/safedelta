"""Does the abstention criterion transport to the external (scPerturBench cellular-context) datasets?

The draft (Results 1) defines the criterion as the CROSS-FITTED evidence: split the target context's
control cells into two halves, build the context correction against one half, score it against the
calibration responses referenced to the other half, swap and average; release iff evidence > 0.
The exact in-sample/leave-one-out convention of the capsule script is not in the handoff, so three
variants are computed and reported side by side:
  evid_loo      leave-one-calibration-out, cross-fitted control   (the strictest reading)
  evid_insample offset from all k calibration responses, scored on the same k, cross-fitted control
  evid_shared   the standard evidence: offset and truth referenced to the SAME control half (naive)
True gain = independent-control PCC(adapted) - PCC(base) on the held-out test perturbations, fold-averaged,
computed from the same pseudobulks with the same seeds/splits/calibration sets as run_w10_trvae.py
(halves_by_key, rng 9000+s, budget 5), so the panels coincide with out/w10_trvae_<ds>.csv rows.
Two bases: pooled_B (a = 0, transparent) and the benchmark's o.o.d. trVAE predictions (from the npz).
Memory: pseudobulk arithmetic only (~1-2 GB per dataset).
"""
import os, sys, argparse
import numpy as np, pandas as pd, anndata as ad
from scipy import stats

SPLIT_KEY = {"sciplex3": "plate", "KaggleCrossCell": "plate_name", "McFarland": "condition3",
             "Haber": "batch"}


def halves_by_key(lab_all, rows, rng):
    lab = lab_all[rows]
    units, cnt = np.unique(lab, return_counts=True)
    order = rng.permutation(len(units))
    units, cnt = units[order], cnt[order]
    a, b, na, nb = [], [], 0, 0
    for u, c in sorted(zip(units, cnt), key=lambda t: -t[1]):
        if na <= nb:
            a.append(u); na += c
        else:
            b.append(u); nb += c
    if not a or not b:
        return None
    ia = rows[np.isin(lab, a)]
    ib = rows[np.isin(lab, b)]
    n = min(len(ia), len(ib))
    return rng.permutation(ia)[:n], rng.permutation(ib)[:n], a, b


def pcc(p, t):
    p = np.asarray(p, np.float64); t = np.asarray(t, np.float64)
    p = p - p.mean(); t = t - t.mean()
    nn = np.linalg.norm(p) * np.linalg.norm(t)
    return float(p @ t / nn) if nn > 0 else np.nan


def run(ds, data_dir, out_dir, budget, splits):
    A = ad.read_h5ad(os.path.join(data_dir, ds + ".h5ad"))
    X = np.asarray(A.X.toarray() if hasattr(A.X, "toarray") else A.X, dtype=np.float32)
    ob = A.obs
    cv = ob["condition1"].astype(str).values; pv = ob["condition2"].astype(str).values
    key = SPLIT_KEY.get(ds)
    cell_mode = key is None
    lab_all = ob[key].astype(str).values if key else None
    conditions = list(ob["condition2"].unique())
    perts = [q for q in conditions if q != "control"]
    ctxs = list(pd.unique(cv))
    T = {}
    for c in ctxs:
        for q in perts:
            m = (cv == c) & (pv == q)
            if m.sum() >= 5:
                T[(c, q)] = X[m].mean(0, dtype=np.float64)
    cfull = {c: X[(cv == c) & (pv == "control")].mean(0, dtype=np.float64) for c in ctxs}
    npz_path = os.path.join(out_dir, f"w10_trvae_{ds}.npz".replace(".npz", "_predictions.npz"))
    P = dict(np.load(npz_path)) if os.path.exists(npz_path) else {}
    rows = []
    for c in ctxs:
        rows_ctrl = np.where((cv == c) & (pv == "control"))[0]
        mine = [q for q in perts if (c, q) in T]
        K = budget if len(mine) >= budget + 1 else max(0, len(mine) - 1)
        if K < 2:
            continue                      # need >= 2 calibration responses for leave-one-out evidence
        pooled = {}
        for q in mine:
            o = [T[(c2, q)] - cfull[c2] for c2 in ctxs if c2 != c and (c2, q) in T]
            if o:
                pooled[q] = np.mean(o, 0)
        for s in range(splits):
            rng = np.random.default_rng(9000 + s)
            if cell_mode:
                pp = rng.permutation(rows_ctrl); h = len(pp) // 2
                hv = (pp[:h], pp[h:2 * h])
            else:
                hv = halves_by_key(lab_all, rows_ctrl, rng)
                if hv is None:
                    continue
            H = [X[hv[0]].mean(0, dtype=np.float64), X[hv[1]].mean(0, dtype=np.float64)]
            n0 = len(hv[0])
            cal = list(rng.permutation(mine)[:K])
            test = [q for q in mine if q not in cal and q in pooled]
            cal = [q for q in cal if q in pooled]
            if len(cal) < 2 or not test:
                continue
            bases = {"pooled_B": {q: (lambda q=q: (lambda fold: pooled[q])) for q in mine}}
            for base_name in ("pooled_B", "trVAE"):
                if base_name == "trVAE" and not all(f"P|{c}|{s}|{f}|{q}" in P for f in (0, 1) for q in cal + test):
                    continue
                def base(q, fold):
                    if base_name == "pooled_B":
                        return pooled[q]
                    return P[f"P|{c}|{s}|{fold}|{q}"] - H[fold]     # trVAE delta prediction of that fold
                ev_loo, ev_ins, ev_sh, tg = [], [], [], {q: [] for q in test}
                for fold in (0, 1):
                    cP, cT = H[fold], H[1 - fold]
                    resid = {q: (T[(c, q)] - cP) - base(q, fold) for q in cal}
                    off_all = np.mean([resid[q] for q in cal], 0)
                    # evidence, leave-one-calibration-out, cross-fitted control
                    g = []
                    for q in cal:
                        off = np.mean([resid[r] for r in cal if r != q], 0)
                        tq = T[(c, q)] - cT
                        g.append(pcc(base(q, fold) + off, tq) - pcc(base(q, fold), tq))
                    ev_loo.append(np.mean(g))
                    # evidence, in-sample (offset from all k), cross-fitted control
                    ev_ins.append(np.mean([pcc(base(q, fold) + off_all, T[(c, q)] - cT) - pcc(base(q, fold), T[(c, q)] - cT) for q in cal]))
                    # naive evidence: same control half on both sides
                    ev_sh.append(np.mean([pcc(base(q, fold) + off_all, T[(c, q)] - cP) - pcc(base(q, fold), T[(c, q)] - cP) for q in cal]))
                    # true gain on held-out test perturbations, independent control
                    for q in test:
                        tq = T[(c, q)] - cT
                        tg[q].append(pcc(base(q, fold) + off_all, tq) - pcc(base(q, fold), tq))
                for q in test:
                    rows.append(dict(dataset=ds, base=base_name, context=c, target=q, split=s, n0=n0, k=len(cal),
                                     evid_loo=float(np.mean(ev_loo)), evid_insample=float(np.mean(ev_ins)),
                                     evid_shared=float(np.mean(ev_sh)), true_gain=float(np.mean(tg[q]))))
    return pd.DataFrame(rows)


def summarise(R):
    out = []
    for (ds, base), g in R.groupby(["dataset", "base"]):
        for ev in ("evid_loo", "evid_insample", "evid_shared"):
            rel = g[ev] > 0
            pos = g.true_gain > 0
            rec = dict(dataset=ds, base=base, evidence=ev, n_panels=len(g), n_contexts=g.context.nunique(),
                       frac_true_positive=pos.mean(), frac_released=rel.mean(),
                       precision=(rel & pos).sum() / max(rel.sum(), 1), recall=(rel & pos).sum() / max(pos.sum(), 1),
                       gain_always=g.true_gain.mean(), gain_gated=(g.true_gain * rel).mean(), gain_never=0.0,
                       harm_avoided=-(g.true_gain[~rel & ~pos]).sum() / len(g),
                       spearman=stats.spearmanr(g[ev], g.true_gain).correlation,
                       spearman_p=stats.spearmanr(g[ev], g.true_gain).pvalue,
                       ctx_spearman_median=g.groupby("context").apply(lambda h: stats.spearmanr(h[ev], h.true_gain).correlation if len(h) > 3 else np.nan).median())
            out.append(rec)
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["sciplex3", "KaggleCrossCell", "McFarland", "Haber"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--splits", type=int, default=3)
    a = ap.parse_args()
    allr = []
    for ds in a.datasets:
        R = run(ds, a.data_dir, a.out_dir, a.budget, a.splits)
        print(f"[{ds}] panels={len(R)} bases={sorted(R.base.unique()) if len(R) else []}", flush=True)
        allr.append(R)
    R = pd.concat(allr, ignore_index=True)
    R.to_csv(os.path.join(a.out_dir, "criterion_external_panels.csv"), index=False)
    S = summarise(R)
    S.to_csv(os.path.join(a.out_dir, "criterion_external_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(S.round(4).to_string(index=False))
