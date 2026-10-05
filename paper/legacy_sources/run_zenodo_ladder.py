"""Equal-marginal-depth, cross-fitted control-regime ladder for the Zenodo-Cellular
datasets that were NOT already analysed from the capsule.

Design (identical to the fixed run_trvae.py / run_scgen.py):
  both regimes use an n/2 control estimate for the PREDICTION and an n/2 control
  estimate for the TRUTH; the prediction array is byte-identical across regimes and
  only the truth's control realisation changes (same vs independent).  Each half
  serves as the prediction side once (cross-fitting), over --splits partitions.

Estimators
  pooled_B         pooled delta from the OTHER contexts            a = 0
  calibration_mean mean of the k calibration deltas                a = 1
  B_plus_offset    pooled_B + (calibration mean - pooled at cal)   a = 1

Emits per (dataset, context, test perturbation, regime, split, fold):
  <m>__pcc  Pearson of the delta          (scale-invariant)
  <m>__dot  centred inner product         (UNNORMALISED; shared-minus-independent
            equals a*sigma^2*(d-1)/n0 and is the only way to separate control
            dependence from predictive strength)
Plus, per (dataset, context): lambda, and a_hat for every estimator.
"""
import argparse, gzip, shutil, os, gc, re, json
import numpy as np, pandas as pd, anndata as ad

CTL = re.compile(r"^(ctrl|control|dmso|dimethyl sulfoxide|vehicle|untreated|"
                 r"unstim(ulated)?|none|no drug|mock|pbs|water|nt|non-targeting)$", re.I)
KEYS = ("plate_name", "plate", "batch", "donor_id", "replicate", "channel",
        "condition3", "time", "well", "sample_id", "patientBatch", "library_id")
MIN_CELLS = 20


def load(name, handoff, tmp):
    p = os.path.join(tmp, name + ".h5ad")
    if not os.path.exists(p):
        with gzip.open(os.path.join(handoff, name + ".h5ad.gz"), "rb") as gi, open(p, "wb") as go:
            shutil.copyfileobj(gi, go, 1 << 22)
    return p


def pick_key(ob, isc, ctxc):
    """Return the measurement-level key giving >=2 groups of >=MIN_CELLS control
    cells in EVERY context, preferring the one with the most cells per group."""
    best = None
    for k in KEYS:
        if k not in ob:
            continue
        g = ob[isc].groupby([ctxc, k], observed=True).size()
        lv = g.groupby(level=0, observed=True).size()
        if len(lv) == ob[ctxc].nunique() and lv.min() >= 2 and g.min() >= MIN_CELLS:
            cand = (float(g.median()), k)
            if best is None or cand > best:
                best = cand
    return best[1] if best else None


def halves_by_key(ob, rows, key, rng):
    """Split the control rows of one context into two disjoint, depth-balanced
    groups of whole measurement units (plates/batches)."""
    lab = ob[key].astype(str).values[rows]
    units, cnt = np.unique(lab, return_counts=True)
    order = rng.permutation(len(units))
    units, cnt = units[order], cnt[order]
    a, b, na, nb = [], [], 0, 0                      # greedy depth balancing
    for u, c in sorted(zip(units, cnt), key=lambda t: -t[1]):
        if na <= nb:
            a.append(u); na += c
        else:
            b.append(u); nb += c
    if not a or not b:
        return None
    ia = rows[np.isin(lab, a)]
    ib = rows[np.isin(lab, b)]
    n = min(len(ia), len(ib))                        # EQUAL MARGINAL DEPTH
    return rng.permutation(ia)[:n], rng.permutation(ib)[:n]


def halves_by_cell(rows, rng):
    p = rng.permutation(rows)
    h = len(p) // 2
    return p[:h], p[h:2 * h]


def zc(v):
    return v - v.mean()


def scores(pred, truth):
    p, t = zc(pred), zc(truth)
    dot = float(p @ t)
    nn = np.linalg.norm(p) * np.linalg.norm(t)
    return (dot / nn if nn > 0 else np.nan), dot


def run_dataset(name, handoff, tmp, budget, splits, out_rows, out_meta):
    p = load(name, handoff, tmp)
    A = ad.read_h5ad(p)
    ob = A.obs
    X = np.asarray(A.X.todense() if hasattr(A.X, "todense") else A.X, dtype=np.float32)
    ctxc = "condition1" if "condition1" in ob else "cell_type"
    pv = ob["perturbation"].astype(str).values
    isc = pd.Series(pv).str.match(CTL).values
    key = pick_key(ob, isc, ctxc)
    ctxs = [c for c in ob[ctxc].astype(str).unique()]
    cv = ob[ctxc].astype(str).values
    perts = sorted(set(pv[~isc]))
    d = X.shape[1]

    # treated pseudobulk per (context, perturbation), and full-control per context
    T = {}
    for c in ctxs:
        for q in perts:
            m = (cv == c) & (pv == q)
            if m.sum() >= 5:
                T[(c, q)] = X[m].mean(0, dtype=np.float64)
    cfull = {c: X[(cv == c) & isc].mean(0, dtype=np.float64) for c in ctxs}

    for c in ctxs:
        rows = np.where((cv == c) & isc)[0]
        mine = [q for q in perts if (c, q) in T]
        if len(mine) < budget + 1:
            continue
        for s in range(splits):
            rng = np.random.default_rng(9000 + s)
            hv = halves_by_key(ob, rows, key, rng) if key else halves_by_cell(rows, rng)
            if hv is None:
                continue
            H = [X[hv[0]].mean(0, dtype=np.float64), X[hv[1]].mean(0, dtype=np.float64)]
            n0 = len(hv[0])
            # lambda: measured shared-control noise-to-signal ratio
            # RAW and CENTRED control-error norms.  The `dot` metric is a CENTRED inner
            # product, so the quantity that must equal dot_gap/a is the CENTRED one;
            # the raw version over-counts by the squared difference in overall mean
            # (i.e. any systematic depth offset between the two control halves).
            dif = H[0] - H[1]
            noi = float(np.sum(dif ** 2) / 2)
            noi_c = float(np.sum(zc(dif) ** 2) / 2)
            dA = np.stack([T[(c, q)] - H[0] for q in mine])
            dB = np.stack([T[(c, q)] - H[1] for q in mine])
            sig = float(np.mean(np.sum(dA * dB, axis=1)))
            cal = list(rng.permutation(mine)[:budget])
            test = [q for q in mine if q not in cal]
            # pooled predictor from the other contexts (never touches this context's control)
            pooled = {}
            for q in test + cal:
                o = [T[(c2, q)] - cfull[c2] for c2 in ctxs if c2 != c and (c2, q) in T]
                if o:
                    pooled[q] = np.mean(o, 0)
            test = [q for q in test if q in pooled]
            if not test:
                continue
            for fold in (0, 1):
                cP, cT_ind = H[fold], H[1 - fold]
                cmean = np.mean([T[(c, q)] - cP for q in cal], 0)
                off = np.mean([(T[(c, q)] - cP) - pooled[q] for q in cal if q in pooled], 0)
                pr = {"pooled_B": lambda q: pooled[q],
                      "calibration_mean": lambda q: cmean,
                      "B_plus_offset": lambda q: pooled[q] + off}
                for regime, cTruth in (("shared", cP), ("independent", cT_ind)):
                    for q in test:
                        row = dict(dataset=name, context=c, target=q, regime=regime,
                                   split=s, fold=fold, budget=budget, n0=n0,
                                   split_level="measurement:" + key if key else "cell")
                        truth = T[(c, q)] - cTruth
                        for m, f in pr.items():
                            pcc, dot = scores(f(q), truth)
                            row[m + "__pcc"] = pcc
                            row[m + "__dot"] = dot
                        out_rows.append(row)
            # a_hat probe: same pipeline, two independent equal-depth controls
            ah = {}
            for m in ("pooled_B", "calibration_mean", "B_plus_offset"):
                P = []
                for cc_ in (H[0], H[1]):
                    cm = np.mean([T[(c, q)] - cc_ for q in cal], 0)
                    of = np.mean([(T[(c, q)] - cc_) - pooled[q] for q in cal if q in pooled], 0)
                    P.append({"pooled_B": pooled[test[0]],
                              "calibration_mean": cm,
                              "B_plus_offset": pooled[test[0]] + of}[m])
                dn = np.linalg.norm(H[0] - H[1])
                ah[m] = float(np.linalg.norm(P[0] - P[1]) / dn) if dn > 0 else np.nan
            out_meta.append(dict(dataset=name, context=c, split=s, n0=n0, d=d,
                                 split_level="measurement:" + key if key else "cell",
                                 n_perturbations=len(mine), lam=noi_c / max(sig, 1e-12),
                                 noise=noi, noise_centred=noi_c, signal=sig,
                                 **{"a_hat_" + k2: v for k2, v in ah.items()}))
    del A, X, ob
    gc.collect()
    os.remove(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--handoff", default="/projects/xunixibao/handoff")
    ap.add_argument("--tmp", default="/projects/xunixibao/handoff/_tmp")
    ap.add_argument("--out", default="zenodo_ladder.csv")
    ap.add_argument("--meta", default="zenodo_lambda.csv")
    A_ = ap.parse_args()
    os.makedirs(A_.tmp, exist_ok=True)
    rows, meta = [], []
    for nm in A_.datasets:
        run_dataset(nm, A_.handoff, A_.tmp, A_.budget, A_.splits, rows, meta)
        print("  [%s] panels=%d contexts=%d" % (nm, len(rows), len(meta)), flush=True)
    pd.DataFrame(rows).to_csv(A_.out, index=False)
    pd.DataFrame(meta).to_csv(A_.meta, index=False)
    print("wrote %s (%d rows) and %s (%d rows)" % (A_.out, len(rows), A_.meta, len(meta)))
