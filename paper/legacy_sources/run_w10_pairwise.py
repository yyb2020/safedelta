"""JOB-NEXT B2: the benchmark's own per-(context, perturbation) models — scPRAM (published #1 on
McFarland) and scGen — under shared vs independent controls, EQUAL MARGINAL DEPTH, CROSS-FITTED.

Model code is copied verbatim from scPerturBench/Cellular_context_generalization/o.o.d./
  myscPRAM.py : SCPRAM(input_dim=n_vars, device='cuda:0'); train_SCPRAM(train, epochs=100);
                predict(train_adata=train, cell_to_pred=c, key_dic, ratio=0.005)
  myscGen.py  : SCGEN.setup_anndata(train, batch_key='condition2', labels_key='condition1');
                SCGEN(train).train(max_epochs=200, batch_size=64, early_stopping=True,
                early_stopping_patience=25); predict(ctrl_key='control', stim_key=q,
                celltype_to_predict=c)
Both scripts build, for each (held-out context c, perturbation q), adata = cells with
condition2 in {q, control} and train = adata minus (c, q).  train therefore contains ALL control
cells of c, and the prediction is generated from them.  Under equal marginal depth the c-control
set inside train is the A half only (B never enters); everything else is unchanged.

Outputs have the same columns as run_w10_trvae.py (see there); the model column is named
`scPRAM` or `scGen`, with `<model>_plus_offset` (residual form) and `<model>_plus_calmean`.
"""
import os, sys, time, argparse, gc
import numpy as np, pandas as pd
import anndata as ad
if not hasattr(ad, "read"):
    ad.read = ad.read_h5ad
import torch
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coupling_probe import probe, lam_meta, append_csv

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


# ------------------------------------------------------------------ model wrappers (verbatim args)
def fit_predict_scpram(train, c, q, epochs):
    from scpram import models
    model = models.SCPRAM(input_dim=train.n_vars, device='cuda:0')
    model = model.to(model.device)
    key_dic = {'condition_key': 'condition2', 'cell_type_key': 'condition1',
               'ctrl_key': 'control', 'stim_key': q, 'pred_key': 'predict'}
    model.train_SCPRAM(train, epochs=epochs)
    pred = model.predict(train_adata=train, cell_to_pred=c, key_dic=key_dic, ratio=0.005)
    X = pred.X.toarray() if hasattr(pred.X, "toarray") else np.asarray(pred.X)
    del model
    return np.asarray(X, np.float64).mean(0)


_SCGEN_SHIMMED = False


def _shim_scgen():
    """scgen 2.1.0 against scvi-tools 0.20.3 (the benchmark image's cpa env) / 1.4.x:
    identical shims to the project's run_scgen.py."""
    global _SCGEN_SHIMMED
    if _SCGEN_SHIMMED:
        return
    import sys, types, typing
    try:
        import scvi._compat  # noqa
    except Exception:
        sys.modules['scvi._compat'] = types.SimpleNamespace(Literal=typing.Literal)
    import scvi.module.base as _smb
    if not hasattr(_smb, "LossRecorder"):
        _smb.LossRecorder = _smb.LossOutput
    import scgen
    from scvi import REGISTRY_KEYS

    def _latent_shim(self, adata=None, batch_size=512, **kw):
        adata = self._validate_anndata(adata)
        dev = next(self.module.parameters()).device
        out = []
        for tensors in self._make_data_loader(adata=adata, batch_size=batch_size):
            x = tensors[REGISTRY_KEYS.X_KEY].to(dev)
            out.append(self.module.inference(x)["qz_m"].detach().cpu().numpy())
        return np.concatenate(out, 0)
    scgen.SCGEN.get_latent_representation = _latent_shim
    _SCGEN_SHIMMED = True


def fit_predict_scgen(train, c, q, epochs):
    _shim_scgen()
    import scgen
    scgen.SCGEN.setup_anndata(train, batch_key="condition2", labels_key="condition1")
    model = scgen.SCGEN(train)
    model.train(max_epochs=epochs, batch_size=64, early_stopping=True, early_stopping_patience=25)
    pred, delta = model.predict(ctrl_key='control', stim_key=q, celltype_to_predict=c)
    X = pred.X.toarray() if hasattr(pred.X, "toarray") else np.asarray(pred.X)
    # scvi-tools keeps every training AnnData alive in class-level manager stores -> ~0.2 GB/model leak
    for store in ("_setup_adata_manager_store", "_per_instance_manager_store"):
        try:
            getattr(scgen.SCGEN, store).clear()
        except Exception:
            pass
    try:
        model._per_instance_manager_store.clear()
    except Exception:
        pass
    del model, pred, delta
    return np.asarray(X, np.float64).mean(0)


FIT = {"scpram": (fit_predict_scpram, 100, "scPRAM"), "scgen": (fit_predict_scgen, 200, "scGen")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=list(FIT))
    ap.add_argument("--data-dir", default="/projects/xunixibao/handoff/_w10/data")
    ap.add_argument("--out-dir", default="/projects/xunixibao/handoff/out")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=0, help="0 = the benchmark's value")
    ap.add_argument("--max-contexts", type=int, default=0)
    ap.add_argument("--contexts", nargs="*", default=None)
    ap.add_argument("--split-mode", default="key", choices=["key", "cell"])
    ap.add_argument("--tag", default="")
    A = ap.parse_args()
    fit, ep_default, MODEL = FIT[A.model]
    epochs = A.epochs or ep_default

    name = A.dataset
    OUT = os.path.join(A.out_dir, f"w10_{A.model}_{name}{A.tag}.csv")
    SENS = OUT.replace(".csv", "_sensitivity.csv")
    META = OUT.replace(".csv", "_meta.csv")
    PRED = OUT.replace(".csv", "_predictions.npz")
    done = set(pd.read_csv(OUT)["context"].astype(str).unique()) if os.path.exists(OUT) else set()
    pred_store = dict(np.load(PRED, allow_pickle=False)) if os.path.exists(PRED) else {}

    adata = ad.read_h5ad(os.path.join(A.data_dir, name + ".h5ad"))
    if "logNor" in adata.layers:
        adata.X = adata.layers["logNor"]          # myscPRAM.py: adata0.X = adata0.layers['logNor']
    adata.X = np.asarray(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, dtype=np.float32)
    ob = adata.obs
    ob["condition1"] = ob["condition1"].astype(str); ob["condition2"] = ob["condition2"].astype(str)
    cv = ob["condition1"].values; pv = ob["condition2"].values
    key = SPLIT_KEY.get(name)
    if key is None or A.split_mode == "cell":
        A.split_mode = "cell"; key = "cell"
    lab_all = ob[key].astype(str).values if key in ob else np.array(["cell"] * adata.n_obs)
    ctxs = list(pd.unique(cv))
    if A.max_contexts:
        ctxs = ctxs[:A.max_contexts]
    if A.contexts:
        ctxs = [c for c in ctxs if c in set(A.contexts)]
    perts = [q for q in pd.unique(pv) if q != "control"]
    X = adata.X; d = X.shape[1]

    T = {}
    for c in pd.unique(cv):
        for q in perts:
            m = (cv == c) & (pv == q)
            if m.sum() >= 5:
                T[(c, q)] = X[m].mean(0, dtype=np.float64)
    cfull = {c: X[(cv == c) & (pv == "control")].mean(0, dtype=np.float64) for c in pd.unique(cv)}
    for (c, q), v in T.items():
        pred_store[f"T|{c}|{q}"] = v

    for c in ctxs:
        if c in done:
            print(f"[{name}/{MODEL}] {c} already done", flush=True); continue
        t0 = time.time()
        rows_ctrl = np.where((cv == c) & (pv == "control"))[0]
        mine = [q for q in perts if (c, q) in T]
        if not mine:
            continue
        K = A.budget if len(mine) >= A.budget + 1 else max(0, len(mine) - 1)
        pooled, trainmean = {}, {}
        for q in mine:
            o = [T[(c2, q)] - cfull[c2] for c2 in pd.unique(cv) if c2 != c and (c2, q) in T]
            if o:
                pooled[q] = np.mean(o, 0)
                trainmean[q] = np.mean([T[(c2, q)] for c2 in pd.unique(cv) if c2 != c and (c2, q) in T], 0)
        rows, srows, mrows = [], [], []
        for s in range(A.splits):
            rng = np.random.default_rng(9000 + s)
            if A.split_mode == "cell":
                pp = rng.permutation(rows_ctrl); h = len(pp) // 2
                hv = (pp[:h], pp[h:2 * h], ["cells"], ["cells"])
            else:
                hv = halves_by_key(lab_all, rows_ctrl, rng)
            if hv is None:
                print(f"[{name}/{MODEL}] {c} split {s}: cannot split by {key}", flush=True); continue
            H = [X[hv[0]].mean(0, dtype=np.float64), X[hv[1]].mean(0, dtype=np.float64)]
            n0 = len(hv[0])
            cal = list(rng.permutation(mine)[:K])
            test = [q for q in mine if q not in cal and q in pooled]
            P_fold = {}
            for fold in (0, 1):
                ia, ib = hv[fold], hv[1 - fold]
                cP, cT_ind = H[fold], H[1 - fold]
                other_B = hv[1 - fold]                      # the B half: excluded from every training set
                Pq = {}
                t1 = time.time()
                for q in mine:
                    # myscPRAM.py / myscGen.py: adata = cells of {q, control}; train = adata minus (c, q)
                    sel = ((pv == q) | (pv == "control"))
                    sel &= ~((cv == c) & (pv == q))
                    sel[other_B] = False                    # equal marginal depth: c-controls in train = A only
                    train = adata[sel].copy()
                    torch.manual_seed(2020); np.random.seed(2020)
                    Pq[q] = fit(train, c, q, epochs)
                    pred_store[f"P|{c}|{s}|{fold}|{q}"] = Pq[q]
                    del train; gc.collect(); torch.cuda.empty_cache()
                pred_store[f"cA|{c}|{s}|{fold}"] = cP
                pred_store[f"idxA|{c}|{s}|{fold}"] = ia
                print(f"[{name}/{MODEL}] {c} split {s} fold {fold}: {len(mine)} models, {time.time()-t1:.0f}s, n0={n0}", flush=True)
                preds = {}
                if K > 0:
                    cmean = np.mean([T[(c, q)] - cP for q in cal], 0)
                    off_B = np.mean([(T[(c, q)] - cP) - pooled[q] for q in cal if q in pooled], 0) if any(q in pooled for q in cal) else None
                    off_V = np.mean([(T[(c, q)] - cP) - (Pq[q] - cP) for q in cal], 0)
                for q in mine:
                    pr = {MODEL: Pq[q] - cP}
                    if K > 0:
                        pr[MODEL + "_plus_offset"] = Pq[q] - cP + off_V
                        pr[MODEL + "_plus_calmean"] = Pq[q] - cP + cmean
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
                        row = dict(dataset=name, model=MODEL, context=c, target=q, regime=regime, split=s, fold=fold,
                                   budget=K, n0=n0, n_control_pred=len(ia), n_control_truth=len(ib),
                                   in_test=q in test, in_cal=q in cal,
                                   split_level=("measurement:" + key) if A.split_mode == "key" else "cell",
                                   arm="ood", epochs_run=epochs)
                        for m, pvec in preds[q].items():
                            pcc, r2, dot = scores(pvec, truth)
                            row[m + "__pcc"] = pcc; row[m + "__r2"] = r2; row[m + "__dot"] = dot
                        rows.append(row)
            if 0 in P_fold and 1 in P_fold:
                for q in mine:
                    for m in P_fold[0][q]:
                        pr = probe(P_fold[0][q][m], P_fold[1][q][m], H[0], H[1])
                        srows.append(dict(dataset=name, model=MODEL, context=c, target=q, split=s, method=m,
                                          n0=n0, in_test=q in test, **pr))
            lm = lam_meta(H[0], H[1], [T[(c, q)] for q in mine])
            mrows.append(dict(dataset=name, model=MODEL, context=c, split=s, n0=n0, d=d,
                              split_level=("measurement:" + key) if A.split_mode == "key" else "cell",
                              plates_A="|".join(hv[2]), plates_B="|".join(hv[3]),
                              budget=K, cal="|".join(cal), test="|".join(test), **lm))
        if rows:
            append_csv(pd.DataFrame(rows), OUT)
        if srows:
            append_csv(pd.DataFrame(srows), SENS)
        if mrows:
            append_csv(pd.DataFrame(mrows), META)
        np.savez(PRED, **pred_store)
        print(f"[{name}/{MODEL}] {c}: {len(mine)} perturbations, {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)
    print("DONE", name, MODEL, flush=True)


if __name__ == "__main__":
    main()
