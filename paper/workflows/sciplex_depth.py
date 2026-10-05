"""Nested control-depth ladder on sciPlex3 — the plate-7 experiment, second deposit.

Follows PROTOCOL_sciplex3_depth_ladder.md, which was frozen before this ran. The scoring
is lifted from handoff/in/run_zenodo_ladder.py (the script that produced the package's
ladder tables) so the estimators and the metric are the same objects, not re-derivations:
centred inner product, three closed-form arms, equal marginal depth, cross-fitting.

What is added here is the ladder itself: control cells are subsampled in a nested sequence
within each half, so a deeper rung contains every shallower one and the only quantity that
moves across rungs is control depth.
"""
import numpy as np, pandas as pd, anndata as ad, re, json

PATH = None
CTL = re.compile(r"^(ctrl|control|dmso|dimethyl sulfoxide|vehicle|untreated|none|mock)$", re.I)
RUNGS = [20, 40, 80, 160, 320, 640]
TREATED = 25
BUDGET = 5
SPLITS = 3


def zc(v):
    return v - v.mean()


def scores(pred, truth):
    p, t = zc(pred), zc(truth)
    dot = float(p @ t)
    nn = np.linalg.norm(p) * np.linalg.norm(t)
    return (dot / nn if nn > 0 else np.nan), dot


def halves_by_plate(plate_lab, rows, rng):
    """Two disjoint, depth-balanced groups of whole plates, truncated to equal size."""
    lab = plate_lab[rows]
    units, cnt = np.unique(lab, return_counts=True)
    a, b, na, nb = [], [], 0, 0
    for u, c in sorted(zip(units, cnt), key=lambda t: -t[1]):
        if na <= nb:
            a.append(u); na += c
        else:
            b.append(u); nb += c
    ia = rows[np.isin(lab, a)]
    ib = rows[np.isin(lab, b)]
    n = min(len(ia), len(ib))
    return rng.permutation(ia)[:n], rng.permutation(ib)[:n]


def main():
    A = ad.read_h5ad(PATH)
    ob = A.obs
    X = np.asarray(A.X.todense() if hasattr(A.X, "todense") else A.X, dtype=np.float32)
    ctx = ob["condition1"].astype(str).values
    pv = ob["perturbation"].astype(str).values
    plate = ob["plate"].astype(str).values
    isc = np.array([bool(CTL.match(x)) for x in pv])
    ctxs = sorted(set(ctx))
    # perturbations with at least TREATED cells in every context
    cnt = pd.crosstab(pv[~isc], ctx[~isc])
    perts = sorted(q for q in cnt.index if (cnt.loc[q] >= TREATED).all())

    rows_out, meta = [], []
    for s in range(SPLITS):
        rng = np.random.default_rng(9000 + s)
        # treated pseudobulk, fixed across the whole ladder for this split
        T = {}
        for c in ctxs:
            for q in perts:
                idx = np.where((ctx == c) & (pv == q))[0]
                T[(c, q)] = X[rng.permutation(idx)[:TREATED]].mean(0, dtype=np.float64)
        # each context's full control mean, for the pooled arm's other contexts
        cfull = {c: X[(ctx == c) & isc].mean(0, dtype=np.float64) for c in ctxs}
        pooled = {}
        for c in ctxs:
            for q in perts:
                o = [T[(c2, q)] - cfull[c2] for c2 in ctxs if c2 != c]
                pooled[(c, q)] = np.mean(o, 0)

        for c in ctxs:
            crows = np.where((ctx == c) & isc)[0]
            ia, ib = halves_by_plate(plate, crows, rng)
            cal = list(rng.permutation(perts)[:BUDGET])
            test = [q for q in perts if q not in cal]
            for m in RUNGS:
                if m > min(len(ia), len(ib)):
                    continue
                # nested: rung m is the first m of the one permutation drawn above
                H = [X[ia[:m]].mean(0, dtype=np.float64), X[ib[:m]].mean(0, dtype=np.float64)]
                dif = H[0] - H[1]
                meta.append(dict(context=c, split=s, n0=m,
                                 noise_centred=float(np.sum(zc(dif) ** 2) / 2),
                                 signal=float(np.mean([np.sum((T[(c, q)] - H[0]) * (T[(c, q)] - H[1]))
                                                       for q in perts]))))
                for fold in (0, 1):
                    cP, cT_ind = H[fold], H[1 - fold]
                    cmean = np.mean([T[(c, q)] - cP for q in cal], 0)
                    off = np.mean([(T[(c, q)] - cP) - pooled[(c, q)] for q in cal], 0)
                    pr = {"pooled_B": lambda q: pooled[(c, q)],
                          "calibration_mean": lambda q: cmean,
                          "B_plus_offset": lambda q: pooled[(c, q)] + off}
                    for q in test:
                        row = dict(context=c, split=s, fold=fold, cells_per_condition=m, drug=q)
                        for name, f in pr.items():
                            p_ = f(q)
                            pcc_s, dot_s = scores(p_, T[(c, q)] - cP)
                            pcc_i, dot_i = scores(p_, T[(c, q)] - cT_ind)
                            rows_out.append(dict(row, method=name,
                                                 pcc_shared=pcc_s, pcc_indep=pcc_i,
                                                 dot_shared=dot_s, dot_indep=dot_i,
                                                 gap=dot_s - dot_i))
        print("split %d done, rows %d" % (s, len(rows_out)), flush=True)

    R = pd.DataFrame(rows_out)
    R["gxn"] = R.gap * R.cells_per_condition
    R.to_csv("sciplex3_depth_numerator_gap.csv", index=False)
    M = pd.DataFrame(meta)
    M["nxn"] = M.noise_centred * M.n0
    M.to_csv("sciplex3_depth_lambda.csv", index=False)
    print("wrote %d rows, %d meta" % (len(R), len(meta)))


if __name__ == "__main__":
    import argparse, os
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--out-dir",required=True)
    args=ap.parse_args()
    PATH=os.path.abspath(args.input)
    os.makedirs(args.out_dir,exist_ok=True)
    os.chdir(args.out_dir)
    main()
