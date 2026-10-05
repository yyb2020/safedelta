"""JOB-W10-A: the benchmark's own trVAE (scPerturBench, Cellular_context_generalization/o.o.d./mytrVAE.py)
scored under two control regimes with EQUAL MARGINAL DEPTH and CROSS-FITTING.

Why trVAE and not CPA: CPA is not one of the 14 methods the benchmark ran in the
cellular-context o.o.d. arm (see Results/Cellular_context_ood/*.csv: CellOT, SCREEN,
baseControl, baseMLP, baseReg, biolord, inVAE, scDisInFact, scGen, scPRAM, scPreGAN,
scVIDR, trVAE, trainMean).  trVAE is, and its o.o.d. script lives in the benchmark's
`cpa` conda environment.  Everything model-related below (early_stopping_kwargs,
get_reconstruction, perturbation_prediction, TRVAE(...) and .train(...) arguments,
the training-set definition) is copied verbatim from mytrVAE.py.

Design per (dataset, held-out context c, split s, fold f):
  the control cells of c are split into two disjoint, depth-balanced groups of whole
  plates (A, B), truncated to equal size.  The benchmark pipeline is run with A as
  ITS ENTIRE control set for c: A is the only c-control in the training set, A is the
  source for perturbation_prediction, and mean(A) is the control subtracted from the
  prediction (their calculateDelta).  The truth is T_q - mean(A) (shared) or
  T_q - mean(B) (independent).  The prediction array is byte-identical across regimes.
  fold 1 swaps A and B (cross-fitting).  Their train_adata in mytrVAE.py includes ALL
  control cells of the held-out context; restricting it to A is the only change, and
  it is what makes B untouched by the prediction pipeline.

Transparent estimators on the identical split (delta space, as in run_zenodo_ladder.py):
  pooled_B          mean over other contexts of (T - c_other)          a = 0
  calibration_mean  mean of the k calibration deltas (T_cal - c_A)     a = 1
  B_plus_offset     pooled_B + mean_cal[(T_cal - c_A) - pooled_B_cal]  a = 1
  trainMean_theirs  the benchmark's own trainMean baseline scored the benchmark's way:
                    mean over other contexts of T (expression space) minus c_A   a = 1
                    (their per-cell Gaussian noise averages out in the pseudobulk)
  trVAE             pseudobulk(perturbation_prediction(A)) - c_A         a = ?
  trVAE_plus_offset trVAE + mean_cal[(T_cal - c_A) - trVAE_cal]           residual form
  trVAE_plus_calmean trVAE + mean_cal(T_cal - c_A)                       run_trvae.py's form

Metrics per row: __pcc (centred Pearson), __r2 (1 - SSE/SST, scale sensitive),
__dot (centred inner product, unnormalised).
Sensitivity probe per (context, split, method, target): a_hat = ||P(A) - P(B)|| / ||c_A - c_B||
(norm ratio, as specified) and a_proj = -<P(A)-P(B), c_A-c_B> / ||c_A-c_B||^2 (projection;
the quantity for which the fold-averaged dot gap equals a * ||c_A - c_B||^2_centred / 2
is an exact identity for ANY predictor).  Only controls + calibration are used.
"""
import os, sys, time, argparse, json, gc
import numpy as np, pandas as pd
import anndata as ad
if not hasattr(ad, "read"):
    ad.read = ad.read_h5ad          # scArches 0.6.1 legacy alias
import torch
import scarches as sca
from scipy.sparse import issparse
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- verbatim from mytrVAE.py
early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss",
    "threshold": 0,
    "patience": 100,
    "reduce_lr": False
}

def get_reconstruction(model, x, encoder_labels=None, decoder_labels = None):

    x_ = torch.log(1 + x)
    if model.model.recon_loss == 'mse':
        x_ = x

    z_mean, z_log_var = model.model.encoder(x_, encoder_labels)
    latent = model.model.sampling(z_mean, z_log_var)
    output = model.model.decoder(latent, decoder_labels)
    return output[0]

def perturbation_prediction(model, adata, source_cond, target_cond):
    device = next(model.model.parameters()).device

    source_adata = adata[adata.obs["condition2"] == source_cond]

    from scarches.dataset.trvae._utils import label_encoder

    encoder_labels = label_encoder(source_adata, model.model.condition_encoder, "condition2")
    decoder_labels = np.zeros_like(encoder_labels) + model.model.condition_encoder[target_cond]

    x = adata.X

    latents = []
    indices = torch.arange(x.shape[0])
    subsampled_indices = indices.split(512)
    for batch in subsampled_indices:
            x_batch = x[batch, :]
            if issparse(x_batch):
                    x_batch = x_batch.toarray()
            x_batch = torch.tensor(x_batch, device=device)
            encoder_labels = torch.tensor(encoder_labels, device=device)
            decoder_labels = torch.tensor(decoder_labels, device=device)
            latent = get_reconstruction(model, x_batch, encoder_labels[batch], decoder_labels[batch])
            latents += [latent.cpu().detach()]

    return np.array(torch.cat(latents))
# ----------------------------------------------------------------------------------------

SPLIT_KEY = {"sciplex3": "plate", "KaggleCrossCell": "plate_name", "McFarland": "condition3", "Haber": "batch"}


def halves_by_key(lab_all, rows, rng):
    """Two disjoint depth-balanced groups of whole measurement units, equal size."""
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


def zc(v):
    return v - v.mean()


def scores(pred, truth):
    p, t = zc(np.asarray(pred, np.float64)), zc(np.asarray(truth, np.float64))
    dot = float(p @ t)
    nn = np.linalg.norm(p) * np.linalg.norm(t)
    pcc = dot / nn if nn > 0 else np.nan
    sst = float(t @ t)
    r2 = 1.0 - float(np.sum((t - p) ** 2)) / sst if sst > 0 else np.nan
    return pcc, r2, dot


def train_trvae(train_adata, conditions, n_epochs):
    """TRVAE(...) and .train(...) arguments verbatim from mytrVAE.py (Kang_OutSample)."""
    trvae = sca.models.TRVAE(adata=train_adata, condition_key="condition2",
                             conditions=conditions, hidden_layer_sizes=[128, 128],
                             recon_loss="mse", dr_rate=0.2,
                             use_bn=True)
    trvae.train(n_epochs=n_epochs, alpha_epoch_anneal=200,
                early_stopping_kwargs=early_stopping_kwargs,
                batch_size=512, clip_value=100,
                monitor=False)
    return trvae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", default="/projects/xunixibao/handoff/_w10/data")
    ap.add_argument("--out-dir", default="/projects/xunixibao/handoff/out")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1000, help="mytrVAE.py: trvae_epochs = 1000")
    ap.add_argument("--max-contexts", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--contexts", nargs="*", default=None, help="restrict to these held-out contexts")
    ap.add_argument("--split-mode", default="key", choices=["key", "cell"], help="cell = random cell-level halves (robustness check)")
    ap.add_argument("--arm", default="ood", choices=["ood", "iid"],
                    help="iid = the benchmark's i.i.d. arm (i.i.d./mytrVAE.py): ONE model trained on iid_test=='train' "
                         "cells of all contexts; predictions from each context's iid_test=='test' control cells; "
                         "truth from its iid_test=='test' treated cells")
    A = ap.parse_args()

    name = A.dataset
    if A.arm == "iid" and not A.tag:
        A.tag = "_iid"
    OUT = os.path.join(A.out_dir, f"w10_trvae_{name}{A.tag}.csv")
    SENS = OUT.replace(".csv", "_sensitivity.csv")
    META = OUT.replace(".csv", "_meta.csv")
    PRED = OUT.replace(".csv", "_predictions.npz")
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["context"].astype(str).unique())
    pred_store = dict(np.load(PRED, allow_pickle=False)) if os.path.exists(PRED) else {}

    adata = ad.read_h5ad(os.path.join(A.data_dir, name + ".h5ad"))
    adata.X = np.asarray(adata.X, dtype=np.float32)          # mytrVAE.py: adata.X.astype(np.float32)
    ob = adata.obs
    cv = ob["condition1"].astype(str).values
    pv = ob["condition2"].astype(str).values
    key = SPLIT_KEY.get(name)
    if key is None:
        A.split_mode = "cell"; key = "cell"
    lab_all = ob[key].astype(str).values if key in ob else np.array(["cell"] * ob.shape[0])
    ctxs = list(pd.unique(cv))
    if A.max_contexts:
        ctxs = ctxs[:A.max_contexts]
    if A.contexts:
        ctxs = [c for c in ctxs if c in set(A.contexts)]
    conditions = list(ob["condition2"].unique())            # mytrVAE.py: conditions = all condition2
    perts = [q for q in conditions if q != "control"]
    X = adata.X
    d = X.shape[1]
    # iid arm: evaluation cells are the benchmark's iid_test == 'test' cells; ood arm: all cells
    ev = (ob["iid_test"].astype(str).values == "test") if A.arm == "iid" else np.ones(adata.n_obs, bool)
    shared_model = None
    if A.arm == "iid":
        train_adata = adata[ob["iid_test"].astype(str).values == "train"].copy()   # i.i.d./mytrVAE.py
        torch.manual_seed(2020); np.random.seed(2020)
        t1 = time.time()
        shared_model = train_trvae(train_adata, conditions, A.epochs)
        print(f"[{name}] iid model trained once on {train_adata.n_obs} cells, {time.time()-t1:.0f}s", flush=True)
        del train_adata; gc.collect()

    # pseudobulks
    T = {}
    for c in pd.unique(cv):
        for q in perts:
            m = (cv == c) & (pv == q) & ev
            if m.sum() >= 5:
                T[(c, q)] = X[m].mean(0, dtype=np.float64)
    cfull = {c: X[(cv == c) & (pv == "control") & ev].mean(0, dtype=np.float64) for c in pd.unique(cv)}
    for c in pd.unique(cv):
        for q in perts:
            if (c, q) in T:
                pred_store[f"T|{c}|{q}"] = T[(c, q)]

    for c in ctxs:
        if c in done:
            print(f"[{name}] {c} already done, skipping", flush=True); continue
        t0 = time.time()
        rows_ctrl = np.where((cv == c) & (pv == "control") & ev)[0]
        mine = [q for q in perts if (c, q) in T]
        if len(mine) < 1:
            print(f"[{name}] {c}: no perturbation with >=5 cells, skip", flush=True); continue
        # few-shot layer needs budget+1 perturbations; otherwise shrink the budget (0 = zero-shot layer only)
        K = A.budget if len(mine) >= A.budget + 1 else max(0, len(mine) - 1)
        if K < A.budget:
            print(f"[{name}] {c}: only {len(mine)} perturbations -> budget {K}", flush=True)
        # pooled (a=0) from the other contexts: never touches c's control
        pooled = {}
        for q in mine:
            o = [T[(c2, q)] - cfull[c2] for c2 in pd.unique(cv) if c2 != c and (c2, q) in T]
            if o:
                pooled[q] = np.mean(o, 0)
        # their trainMean information: other contexts' treated mean in EXPRESSION space
        trainmean = {}
        for q in mine:
            o = [T[(c2, q)] for c2 in pd.unique(cv) if c2 != c and (c2, q) in T]
            if o:
                trainmean[q] = np.mean(o, 0)
        rows, srows, mrows = [], [], []
        for s in range(A.splits):
            rng = np.random.default_rng(9000 + s)
            if A.split_mode == "cell":
                pp = rng.permutation(rows_ctrl); h = len(pp) // 2
                hv = (pp[:h], pp[h:2 * h], ["cells"], ["cells"])
            else:
                hv = halves_by_key(lab_all, rows_ctrl, rng)
            if hv is None:
                print(f"[{name}] {c} split {s}: cannot split by {key}", flush=True); continue
            H = [X[hv[0]].mean(0, dtype=np.float64), X[hv[1]].mean(0, dtype=np.float64)]
            n0 = len(hv[0])
            dif = H[0] - H[1]
            noi_c = float(np.sum(zc(dif) ** 2) / 2)
            dA = np.stack([T[(c, q)] - H[0] for q in mine]); dB = np.stack([T[(c, q)] - H[1] for q in mine])
            sig = float(np.mean(np.sum(dA * dB, axis=1)))
            cal = list(rng.permutation(mine)[:K])
            test = [q for q in mine if q not in cal and q in pooled]
            P_fold = {}
            ep_used = {}
            for fold in (0, 1):
                ia, ib = hv[fold], hv[1 - fold]
                cP, cT_ind = H[fold], H[1 - fold]
                # ---- the benchmark's pipeline, with A as the ENTIRE control set of context c
                adata_source = adata[ia].copy()
                t1 = time.time()
                if shared_model is None:
                    keep = (cv != c)
                    keep[ia] = True
                    train_adata = adata[keep].copy()
                    torch.manual_seed(2020); np.random.seed(2020)
                    trvae = train_trvae(train_adata, conditions, A.epochs)
                else:
                    trvae = shared_model; train_adata = None
                try:
                    ep_used[fold] = int(trvae.trainer.epoch) if hasattr(trvae, "trainer") else -1
                except Exception:
                    ep_used[fold] = -1
                Pq = {}
                for q in mine:
                    corrected = perturbation_prediction(trvae, adata_source, "control", q)   # verbatim
                    Pq[q] = np.asarray(corrected, np.float64).mean(0)
                    pred_store[f"P|{c}|{s}|{fold}|{q}"] = Pq[q]
                pred_store[f"cA|{c}|{s}|{fold}"] = cP
                pred_store[f"idxA|{c}|{s}|{fold}"] = ia
                if shared_model is None:
                    del trvae, train_adata
                gc.collect(); torch.cuda.empty_cache()
                print(f"[{name}] {c} split {s} fold {fold}: trained {time.time()-t1:.0f}s, epochs={ep_used[fold]}, n0={n0}", flush=True)
                # ---- estimators (delta space); prediction-side control is cP for everything
                preds = {}
                if K > 0:
                    cmean = np.mean([T[(c, q)] - cP for q in cal], 0)
                    off_B = np.mean([(T[(c, q)] - cP) - pooled[q] for q in cal if q in pooled], 0) if any(q in pooled for q in cal) else None
                    off_V = np.mean([(T[(c, q)] - cP) - (Pq[q] - cP) for q in cal], 0)
                for q in mine:
                    pr = {"trVAE": Pq[q] - cP}
                    if K > 0:
                        pr["trVAE_plus_offset"] = Pq[q] - cP + off_V
                        pr["trVAE_plus_calmean"] = Pq[q] - cP + cmean
                        pr["calibration_mean"] = cmean
                    if q in pooled:
                        pr["pooled_B"] = pooled[q]
                        pr["trainMean_theirs"] = trainmean[q] - cP
                        if K > 0 and off_B is not None:
                            pr["B_plus_offset"] = pooled[q] + off_B
                    preds[q] = pr
                P_fold[fold] = preds
                for regime, cTruth in (("shared", cP), ("independent", cT_ind)):
                    for q in mine:
                        truth = T[(c, q)] - cTruth
                        row = dict(dataset=name, context=c, target=q, regime=regime, split=s, fold=fold,
                                   budget=K, n0=n0, n_control_pred=len(ia), n_control_truth=len(ib),
                                   in_test=q in test, in_cal=q in cal, split_level=("measurement:" + key) if A.split_mode == "key" else "cell",
                                   arm=A.arm,
                                   plates_pred="|".join(hv[2] if fold == 0 else hv[3]),
                                   epochs_run=ep_used[fold])
                        for m, pvec in preds[q].items():
                            pcc, r2, dot = scores(pvec, truth)
                            row[m + "__pcc"] = pcc; row[m + "__r2"] = r2; row[m + "__dot"] = dot
                        rows.append(row)
            # ---- sensitivity probe: same pipeline, two independent equal-depth controls
            if 0 in P_fold and 1 in P_fold:
                dvec = H[0] - H[1]
                dn = float(np.linalg.norm(dvec)); dn_c = float(np.linalg.norm(zc(dvec)))
                for q in mine:
                    for m in P_fold[0][q]:
                        dp = P_fold[0][q][m] - P_fold[1][q][m]
                        srows.append(dict(dataset=name, context=c, target=q, split=s, method=m, n0=n0,
                                          in_test=q in test,
                                          a_hat=float(np.linalg.norm(dp) / dn),
                                          a_proj=float(-(dp @ dvec) / dn ** 2),
                                          a_proj_centred=float(-(zc(dp) @ zc(dvec)) / dn_c ** 2),
                                          cos=float(-(dp @ dvec) / (np.linalg.norm(dp) * dn)) if np.linalg.norm(dp) > 0 else np.nan,
                                          dc=dn, dc_centred=dn_c))
            mrows.append(dict(dataset=name, context=c, split=s, n0=n0, d=d, split_level=("measurement:" + key) if A.split_mode == "key" else "cell",
                              plates_A="|".join(hv[2]), plates_B="|".join(hv[3]),
                              n_perturbations=len(mine), cal="|".join(cal), test="|".join(test),
                              lam=noi_c / max(sig, 1e-12), noise_centred=noi_c, signal=sig,
                              epochs_fold0=ep_used.get(0, -1), epochs_fold1=ep_used.get(1, -1)))
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        if srows:
            pd.DataFrame(srows).to_csv(SENS, mode="a", header=not os.path.exists(SENS), index=False)
        pd.DataFrame(mrows).to_csv(META, mode="a", header=not os.path.exists(META), index=False)
        np.savez(PRED, **pred_store)
        print(f"[{name}] {c}: {len(mine)} perturbations, {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)
    print("DONE", name, flush=True)


if __name__ == "__main__":
    main()
