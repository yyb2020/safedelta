"""Freeze the two gene sets the manuscript cites but no table carried, so that both
become recomputable inside the package instead of standing as NOT-RECOMPUTABLE.

  plate7_gene_axis.csv      d = 7,007 -- the genes scored in the depth experiment.
                            Taken from the depth-sweep archive's own gene array; the
                            pseudobulk archive carries the SAME SET in a different
                            order (verified: set equality, 7,004 of 7,007 positions
                            differ), so d is unambiguous.
  spatial_expressed_pool.csv  12,106 -- the expressed-gene pool of the spatial data,
                            the set over which a liver set score is formed. Definition
                            as published: mean log1p-CP10k across all spots of all
                            sections > 0.02.

Run:  python3 04_code/freeze_gene_sets.py
Requires the raw spatial archive; it extracts to /tmp/g525 if the matrices are absent.
"""
import numpy as np, pandas as pd, glob, os, sys, subprocess, h5py
from scipy import sparse as spr

SUB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TAR = "/projects/xunixibao/files_new/raw/datasets/GSE338525_RAW.tar"
SPOT = "/projects/xunixibao/files_new/results/gse338525-external-spatial-v1/spot-scores.tsv.gz"
THR = 0.02

os.makedirs("/tmp/g525", exist_ok=True)
if len(glob.glob("/tmp/g525/*.h5")) < 3:
    subprocess.run(["tar", "-xf", TAR, "--wildcards", "*filtered_feature_bc_matrix.h5"],
                   cwd="/tmp/g525", check=True)
h5s = sorted(glob.glob("/tmp/g525/*.h5"))
sp = pd.read_csv(SPOT, sep="\t")
f2s = {}
for f in h5s:
    b = os.path.basename(f).split("_")
    f2s["%s_%s" % (b[1], b[2])] = f
SAMP = [s for s in sp["sample"].unique() if s in f2s]

names, tot, nsp = None, None, 0
for s in SAMP:
    with h5py.File(f2s[s], "r") as g:
        m = g["matrix"]; sh = m["shape"][:]
        M = spr.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=(sh[0], sh[1]))
        gn = np.array([x.decode() for x in m["features"]["id"][:]])
    if names is None:
        names, tot = gn, np.zeros(len(gn))
    assert list(gn) == list(names), "gene order differs between sections: %s" % s
    A = M.T.tocsr().astype(np.float32)                      # spots x genes, sparse
    lib = np.asarray(A.sum(1)).ravel()
    A = A.multiply((1e4 / np.maximum(lib, 1))[:, None]).tocsr()
    A.data = np.log1p(A.data)                               # log1p of CP10k, sparse-safe
    tot += np.asarray(A.sum(0)).ravel()
    nsp += A.shape[0]
    del M, A

gm = tot / nsp
pool = np.where(gm > THR)[0]
pd.DataFrame({"gene": names[pool], "mean_log1p_cp10k": gm[pool].round(6)}).to_csv(
    os.path.join(SUB, "03_data", "spatial_expressed_pool.csv"), index=False)
print("sections %d | spots %d | genes measured %d" % (len(SAMP), nsp, len(names)))
print("expressed pool at mean log1p-CP10k > %.2f : %d genes" % (THR, len(pool)))
print("sensitivity of the pool size to the threshold:")
for t in (0.01, 0.02, 0.05, 0.1):
    print("   > %.2f -> %6d genes" % (t, int((gm > t).sum())))
