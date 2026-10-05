"""Threshold sensitivity on the EXACT published definition.

The published split is defined on the 438-gene symbol-mapped table of
pHSC-induced genes (liver_adapter_decomposition.csv, all pHSC log2FC > 1):
    CORE          = pHSC > 1  AND  LX-2 > 1                  -> 94   (verified)
    pHSC-specific = pHSC > 1  AND  |LX-2| < 0.25             -> 146  (verified)

Because that table is itself filtered at pHSC > 1, the induction cutoff can only
be swept UPWARD from 1. The downward direction needs the ENSG->symbol map used
to build the table, which is not in the handoff package; that limitation is
reported rather than papered over with a different gene pool.

Scoring follows the published convention recovered from the original session:
within each section every pool gene (mean log1p-CP10k > 0.02) is z-scored across
that section's spots; a set score is the mean z over the set's genes; the
reported statistic is median(scar spots) - median(hepatocyte spots).
"""
import numpy as np, pandas as pd, glob, os, h5py
from scipy import sparse as spr
from scipy.stats import wilcoxon

SPOT = "/projects/xunixibao/files_new/results/gse338525-external-spatial-v1/spot-scores.tsv.gz"
RNG = np.random.default_rng(0)

D = pd.read_csv("liver_adapter_decomposition.csv")
assert (D.pHSC > 1).all() and len(D) == 438
sig = pd.read_csv("liver_core94_signature.csv")
assert int(((D.pHSC > 1) & (D.LX2 > 1)).sum()) == 94
assert int(((D.pHSC > 1) & (D.LX2.abs() < 0.25)).sum()) == 146
assert set(D.ensg[(D.pHSC > 1) & (D.LX2 > 1)]) == set(sig.ensg)
print("published definition verified on the 438-gene table: CORE 94 (identical to saved), pHSC-specific 146")

sp = pd.read_csv(SPOT, sep="\t")
f2s = {}
for f in sorted(glob.glob("/tmp/g525/*.h5")):
    b = os.path.basename(f).split("_"); f2s["%s_%s" % (b[1], b[2])] = f
SAMP = [s for s in sp["sample"].unique() if s in f2s]

ZS, names, tot, nsp = {}, None, None, 0
for s in SAMP:
    with h5py.File(f2s[s], "r") as g:
        m = g["matrix"]; sh = m["shape"][:]
        M = spr.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=(sh[0], sh[1]))
        bc = np.array([x.decode() for x in m["barcodes"][:]])
        ids = np.array([x.decode().split(".")[0] for x in m["features"]["id"][:]])
    if names is None:
        names = ids; tot = np.zeros(len(ids))
    C = M.T.tocsr().astype(np.float32)
    C = C.multiply((1e4 / np.maximum(np.asarray(C.sum(1)).ravel(), 1))[:, None]).tocsr()
    C.data = np.log1p(C.data)
    tot += np.asarray(C.sum(0)).ravel(); nsp += C.shape[0]
    dd = sp[sp["sample"] == s].set_index("spot_id")
    c = dd.index.intersection(pd.Index(bc)); bi = {b: i for i, b in enumerate(bc)}
    rr = np.array([bi[x] for x in c]); dd = dd.loc[c]
    ZS[s] = (C[rr].tocsc(), (dd.region == "Scar").values, (dd.region == "Hep").values)
    del M, C
gm = tot / nsp
pool = np.where(gm > 0.02)[0]
gi = {g: i for i, g in enumerate(names)}
p2r = {int(g): i for i, g in enumerate(pool)}
MS = {}
for s in SAMP:
    Z, _, _ = ZS[s]; n = Z.shape[0]; Zp = Z[:, pool]
    mu = np.asarray(Zp.sum(0)).ravel() / n
    sq = Zp.copy(); sq.data = sq.data ** 2
    sd = np.sqrt(np.maximum(np.asarray(sq.sum(0)).ravel() / n - mu ** 2, 0))
    MS[s] = (mu, np.where(sd > 0, sd, 1.0)); del Zp, sq
print("spatial: %d sections, %d spots, pool %d genes" % (len(SAMP), nsp, len(pool)))

def per_section(cols):
    pc = np.array([p2r[int(c)] for c in cols if int(c) in p2r])
    if len(pc) == 0:
        return pd.Series(dtype=float), 0
    out = {}
    for s in SAMP:
        Z, isc, ish = ZS[s]
        if isc.sum() < 10 or ish.sum() < 10:
            continue
        mu, sd = MS[s]
        v = ((np.asarray(Z[:, pool[pc]].todense(), dtype=np.float32) - mu[pc]) / sd[pc]).mean(1)
        out[s] = float(np.median(v[isc]) - np.median(v[ish]))
    return pd.Series(out), len(pc)

dec = pd.qcut(gm[pool], 10, labels=False, duplicates="drop")
decs = pd.Series(dec, index=pool)
def matched_null(cols, ndraw=300):
    want = decs.loc[[c for c in cols if c in p2r]].value_counts()
    vals = []
    for _ in range(ndraw):
        pick = []
        for dv, k in want.items():
            cand = pool[dec == dv]
            pick.extend(RNG.choice(cand, size=min(k, len(cand)), replace=False))
        vals.append(per_section(np.array(pick))[0].median())
    return np.array(vals)

rows, secs = [], []
for t in (1.0, 1.25, 1.5, 2.0):
    cg = D.ensg[(D.pHSC > t) & (D.LX2 > t)]
    sg = D.ensg[(D.pHSC > t) & (D.LX2.abs() < 0.25)]
    ci = [gi[g] for g in cg if g in gi]; si_ = [gi[g] for g in sg if g in gi]
    rc, nc_m = per_section(np.array(ci)); rs, ns_m = per_section(np.array(si_))
    nullc, nulls = matched_null(ci), matched_null(si_)
    common = rc.index.intersection(rs.index); dif = rc[common] - rs[common]
    rows.append(dict(threshold=t, n_core=len(cg), n_core_measurable=nc_m,
                     n_spec=len(sg), n_spec_measurable=ns_m,
                     core=rc.median(), core_pos=int((rc > 0).sum()), core_n=len(rc),
                     core_P=float((nullc >= rc.median()).mean() + 1 / len(nullc)),
                     spec=rs.median(), spec_pos=int((rs > 0).sum()),
                     spec_P=float((nulls >= rs.median()).mean() + 1 / len(nulls)),
                     paired=dif.median(), paired_pos=int((dif > 0).sum()), paired_n=len(dif),
                     paired_P=float(wilcoxon(dif, alternative="greater")[1])))
    secs.append(pd.DataFrame({"section": rc.index, "core": rc.values,
                              "spec": rs.reindex(rc.index).values, "threshold": t}))
    r = rows[-1]
    print("t=%.2f | CORE %3d genes (%3d measurable) %+.4f  %2d/%d pos  P=%.4f"
          " | SPEC %3d (%3d) %+.4f  P=%.4f | paired %+.4f  %2d/%d  P=%.2e"
          % (t, r["n_core"], nc_m, r["core"], r["core_pos"], r["core_n"], r["core_P"],
             r["n_spec"], ns_m, r["spec"], r["spec_P"], r["paired"], r["paired_pos"],
             r["paired_n"], r["paired_P"]))

S = pd.DataFrame(rows)
S.round(6).to_csv("liver_threshold_sensitivity.csv", index=False)
pd.concat(secs).round(6).to_csv("liver_threshold_sensitivity_per_section.csv", index=False)
print("\npublished t=1 row for comparison: CORE +0.333 (15/15, P=0.003) | SPEC +0.103 (P=0.209) | paired +0.249 (15/15, P=3e-5)")
