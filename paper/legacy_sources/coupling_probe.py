"""Shared code for the control-coupling probe, the lambda meta record and safe CSV appends.
Used by run_trvae.py, run_scgen.py and run_w10_trvae.py so all three emit identical columns.

a_proj_centred is the coefficient for which, after cross-fitting over the two folds,
    mean_fold(dot_shared - dot_independent) == a_proj_centred * ||zc(c0 - c1)||^2 / 2
holds as an algebraic identity for ANY predictor (dot = centred inner product).
a_hat (norm ratio) equals a_proj only for predictors that are linear in the control;
for non-linear pipelines a_hat = a_proj / cos and over-states the coupling.
"""
import os
import numpy as np, pandas as pd


def zc(v):
    v = np.asarray(v, np.float64)
    return v - v.mean()


def probe(P0, P1, c0, c1):
    """P0, P1: prediction vectors (delta space) from the two folds; c0, c1: the two control
    estimates.  Returns dict with a_hat, a_proj, a_proj_centred, cos, dc, dc_centred."""
    dp = np.asarray(P0, np.float64) - np.asarray(P1, np.float64)
    dvec = np.asarray(c0, np.float64) - np.asarray(c1, np.float64)
    dn = float(np.linalg.norm(dvec)); dn_c = float(np.linalg.norm(zc(dvec)))
    npd = float(np.linalg.norm(dp))
    return dict(a_hat=npd / dn if dn > 0 else np.nan,
                a_proj=float(-(dp @ dvec) / dn ** 2) if dn > 0 else np.nan,
                a_proj_centred=float(-(zc(dp) @ zc(dvec)) / dn_c ** 2) if dn_c > 0 else np.nan,
                cos=float(-(dp @ dvec) / (npd * dn)) if npd > 0 and dn > 0 else np.nan,
                dc=dn, dc_centred=dn_c)


def lam_meta(H0, H1, treated_pseudobulks):
    """lambda = centred shared-control noise / cross-half signal, as in run_w10_trvae.py.
    treated_pseudobulks: list of T_q vectors (all perturbations of the context with data)."""
    H0 = np.asarray(H0, np.float64); H1 = np.asarray(H1, np.float64)
    dif = H0 - H1
    noi = float(np.sum(dif ** 2) / 2)
    noi_c = float(np.sum(zc(dif) ** 2) / 2)
    dA = np.stack([np.asarray(T, np.float64) - H0 for T in treated_pseudobulks])
    dB = np.stack([np.asarray(T, np.float64) - H1 for T in treated_pseudobulks])
    sig = float(np.mean(np.sum(dA * dB, axis=1)))
    return dict(lam=noi_c / max(sig, 1e-12), noise=noi, noise_centred=noi_c, signal=sig,
                n_perturbations=len(treated_pseudobulks))


def append_csv(df, path):
    """Append rows to a CSV whose header may differ (older runs lacking new columns).
    Aligns to the existing header; if new columns appear, rewrites the file with the union."""
    if not os.path.exists(path):
        df.to_csv(path, index=False); return
    old_cols = list(pd.read_csv(path, nrows=0).columns)
    if set(df.columns) <= set(old_cols):
        df.reindex(columns=old_cols).to_csv(path, mode="a", header=False, index=False)
    else:
        old = pd.read_csv(path)
        pd.concat([old, df], ignore_index=True, sort=False).to_csv(path, index=False)
