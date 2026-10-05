"""scGen under shared vs independent controls — does the ranking invert?

For each held-out cell line c:
  train scGen on all other lines (control + all drugs) + line c's control + k calibration drugs
  predict c's response to each held-out test drug via latent vector arithmetic
  score the same prediction twice: shared control vs independent control halves
Compares scGen against the three transparent estimators used in the main analysis.
"""
import sys, time, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coupling_probe import probe, lam_meta, append_csv
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, torch
# scgen 2.1.0 imports scvi.module.base.LossRecorder, removed in scvi-tools >=1.1.
# LossOutput takes the same first three positional args (loss, reconstruction_loss, kl_local).
import scvi.module.base as _smb
if not hasattr(_smb, "LossRecorder"):
    _smb.LossRecorder = _smb.LossOutput
import scgen

from scvi import REGISTRY_KEYS

def _latent_shim(self, adata=None, batch_size=512, **kw):
    """scgen 2.1.0's module returns qz_m/qz_v (old scvi names); scvi-tools 1.4.2's
    VAEMixin.get_latent_representation expects qzm/qzv. Read the module directly
    and return the posterior mean, which is what scGen's arithmetic uses."""
    adata = self._validate_anndata(adata)
    dev = next(self.module.parameters()).device
    out = []
    for tensors in self._make_data_loader(adata=adata, batch_size=batch_size):
        x = tensors[REGISTRY_KEYS.X_KEY].to(dev)
        out.append(self.module.inference(x)["qz_m"].detach().cpu().numpy())
    return np.concatenate(out, 0)

scgen.SCGEN.get_latent_representation = _latent_shim

N_LINES = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SPLITS  = int(sys.argv[3]) if len(sys.argv) > 3 else 3
K_CAL = 5
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
rng = np.random.default_rng(0)

Z = np.load("plate7_cells_subset.npz", allow_pickle=True)
X = Z["X"]; grp = Z["group"]
line_of = np.array([str(x) for x in Z["line"]]); drug_of = np.array([str(x) for x in Z["drug"]])
genes = np.array([str(x) for x in Z["genes"]])
cell_line = line_of[grp]; cond = drug_of[grp]
print("cells %d | genes %d | lines %d | drugs %d" % (X.shape[0], X.shape[1],
      len(set(cell_line)), len(set(cond)) - 1), flush=True)

A = ad.AnnData(X=X.astype(np.float32),
               obs=pd.DataFrame({"cell_line": cell_line, "condition": cond},
                                index=[f"c{i}" for i in range(X.shape[0])]),
               var=pd.DataFrame(index=genes))
sc.pp.normalize_total(A, target_sum=1e4); sc.pp.log1p(A)

LINES = sorted(set(cell_line)); DRUGS = sorted(d for d in set(cond) if d != "CONTROL")
# two independent control halves per line
half = np.zeros(A.n_obs, dtype=int)
for L in LINES:
    m = np.where((cell_line == L) & (cond == "CONTROL"))[0]
    h = rng.permutation(len(m)); half[m[h[:len(m)//2]]] = 1; half[m[h[len(m)//2:]]] = 2
A.obs["half"] = half

def _scores(pred_raw, truth_raw):
    """PCC (scale-invariant), R2 of the delta (scale-sensitive), and the raw centred inner product.
    Under PCC a 'baseline x fitted scalar' comparator is mathematically identical to the
    baseline, so the magnitude-calibration critique (Genes 2026, 10.3390/genes17070816)
    can only be answered with a scale-sensitive metric. Both are emitted per panel."""
    p = np.asarray(pred_raw, float); t = np.asarray(truth_raw, float)
    pc = p - p.mean(); tc = t - t.mean()
    pcc = float(pc @ tc / max(np.linalg.norm(pc) * np.linalg.norm(tc), 1e-12))
    ss = max(float(((t - t.mean()) ** 2).sum()), 1e-12)
    r2 = float(1.0 - ((p - t) ** 2).sum() / ss)
    # UNNORMALISED centred inner product. Its shared-minus-independent difference equals
    # a * sigma^2 * d / n0 exactly (verified in simulation to 5e-4), where a is how much of the
    # control estimate the prediction carries. Nicol et al.'s Prop. 1 is the a = 1 case, where the
    # r^2 dependence enters only through the NORMALISATION. Reporting this raw gap alongside the
    # PCC gap is what separates "the method leans on the target control" (numerator, a) from
    # "the method's prediction is weak" (denominator, r^2) — the two are confounded in PCC alone.
    return pcc, r2, float(pc @ tc)


def _fit_scalar(basis_deltas, obs_deltas):
    """s = argmin_s sum_p ||obs_p - s*basis_p||^2, fitted on calibration perturbations only."""
    num = sum(float(b @ o) for b, o in zip(basis_deltas, obs_deltas))
    den = sum(float(b @ b) for b in basis_deltas)
    return num / max(den, 1e-12)


def zr(v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)

OUT = "scgen_control_regime.csv"
import os
done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    done = set(prev.line.unique())
    print("resuming; already done: %d lines" % len(done), flush=True)

rows = []
for li, C in enumerate(LINES[:N_LINES]):
    if C in done:
        continue
    t0 = time.time()
    cal = list(rng.choice(DRUGS, K_CAL, replace=False))
    test = [d for d in DRUGS if d not in cal]
    drop = (A.obs.cell_line.values == C) & np.isin(A.obs.condition.values, test)
    train = A[~drop].copy()
    # scGen convention: batch_key = the perturbation condition, labels_key = the cell type
    scgen.SCGEN.setup_anndata(train, batch_key="condition", labels_key="cell_line")
    model = scgen.SCGEN(train)
    model.train(max_epochs=EPOCHS, batch_size=256, early_stopping=True,
                early_stopping_patience=10, accelerator="gpu", devices=1)
    obs = A.obs
    isC = obs.cell_line.values == C
    ctrlA = np.where(isC & (obs.condition.values == "CONTROL") & (obs.half.values == 1))[0]
    ctrlB = np.where(isC & (obs.condition.values == "CONTROL") & (obs.half.values == 2))[0]
    ctrlALL = np.concatenate([ctrlA, ctrlB])
    others = ~isC
    Xd = A.X
    others_mask = others
    pooled = {d: np.asarray(Xd[others_mask & (obs.condition.values == d)]).mean(0)
                 - np.asarray(Xd[others_mask & (obs.condition.values == "CONTROL")]).mean(0) for d in DRUGS}
    other_lines = [L for L in LINES if L != C]

    def sc_predict(stim, basis_idx):
        # restrict the latent arithmetic to lines that actually have this drug, otherwise
        # scGen's internal per-cell-line balancer hits the held-out line's empty stim set
        out = model.predict(ctrl_key="CONTROL", stim_key=stim,
                            adata_to_predict=A[basis_idx].copy(),
                            restrict_arithmetic_to={"cell_line": other_lines})
        p = out[0] if isinstance(out, tuple) else out
        Xp = p.X if hasattr(p, "X") else p
        return np.asarray(Xp).mean(0)

    test_ok = [d for d in test if (isC & (obs.condition.values == d)).sum() >= 20]
    obs_expr = {d: np.asarray(Xd[isC & (obs.condition.values == d)]).mean(0) for d in test_ok}
    obs_all = {d: np.asarray(Xd[isC & (obs.condition.values == d)]).mean(0) for d in DRUGS
               if (isC & (obs.condition.values == d)).sum() >= 20}
    cal_ok = [d for d in cal if d in obs_all]

    # EQUAL-MARGINAL-DEPTH, CROSS-FITTED DESIGN — see run_trvae.py for the full rationale.
    # Both regimes use an n/2 control for the prediction AND an n/2 control for the truth, so
    # marginal control precision is identical; the prediction array is the same in both regimes
    # and only the truth's control realization changes. Each half serves as prediction side once,
    # over ARGS.splits random disjoint partitions. scGen training is split-independent.
    sens_store, ctrl_store = {}, {}
    for sp in range(SPLITS):
        rs = np.random.default_rng(1000 + sp)
        perm = rs.permutation(len(ctrlALL))
        halves = [ctrlALL[perm[:len(perm)//2]], ctrlALL[perm[len(perm)//2:]]]
        for fold in (0, 1):
            ip, it = halves[fold], halves[1-fold]
            cP = np.asarray(Xd[ip]).mean(0); cT = np.asarray(Xd[it]).mean(0)
            cal_mean = np.mean([np.asarray(Xd[isC & (obs.condition.values == d)]).mean(0) - cP
                                for d in cal], 0)
            pooled_cal = {d: np.asarray(Xd[others_mask & (obs.condition.values == d)]).mean(0)
                             - np.asarray(Xd[others_mask & (obs.condition.values == "CONTROL")]).mean(0)
                          for d in cal}
            cal_obs = [np.asarray(Xd[isC & (obs.condition.values == d)]).mean(0) - cP for d in cal]
            s_hat = _fit_scalar([pooled_cal[d] for d in cal], cal_obs)
            # residual-form context offset: mean over calibration drugs of [(T_cal - cP) - (sg_cal - cP)]
            sg_cal = {}
            for d in cal_ok:
                try:
                    sg_cal[d] = sc_predict(d, ip)
                except Exception as e:
                    print("    cal predict failed %s %s: %r" % (C, d, e), flush=True)
            off_V = (np.mean([(obs_all[d] - cP) - (sg_cal[d] - cP) for d in sg_cal], 0)
                     if sg_cal else np.zeros_like(cP))
            for d in test_ok:
                try:
                    sg = sc_predict(d, ip)
                except Exception as e:
                    print("    predict failed %s %s: %r" % (C, d, e), flush=True); continue
                preds = {"scGen": sg - cP,
                         "scGen_plus_offset": sg - cP + off_V,        # residual form, a ~ 1
                         "scGen_plus_calmean": sg - cP + cal_mean,    # additive form (old name scGen_plus_offset)
                         "pooled_B": pooled[d],
                         "pooled_B_scaled": s_hat * pooled[d],
                         "calibration_mean": cal_mean,
                         "B_plus_offset": pooled[d] + cal_mean}
                for regime, ctrue in (("shared", cP), ("independent", cT)):
                    truth = obs_expr[d] - ctrue
                    row = dict(line=C, drug=d, regime=regime, split=sp, fold=fold,
                               n_control_pred=len(ip), n_control_truth=len(it), s_hat=float(s_hat))
                    for nm, pv in preds.items():
                        pcc, r2, dot = _scores(pv, truth)
                        row[nm] = pcc; row[nm + "__r2"] = r2; row[nm + "__dot"] = dot
                    rows.append(row)
                sens_store.setdefault((sp, fold), {})[d] = {k_: np.asarray(v_, float) for k_, v_ in preds.items()}
            ctrl_store[(sp, fold)] = cP

    # SENSITIVITY PROBE (free) — see run_trvae.py for the rationale.
    srows, mrows = [], []
    for sp in range(SPLITS):
        if (sp, 0) not in ctrl_store or (sp, 1) not in ctrl_store:
            continue
        common = set(sens_store.get((sp, 0), {})) & set(sens_store.get((sp, 1), {}))
        for d in common:
            for m in sens_store[(sp, 0)][d]:
                pr = probe(sens_store[(sp, 0)][d][m], sens_store[(sp, 1)][d][m],
                           ctrl_store[(sp, 0)], ctrl_store[(sp, 1)])
                srows.append(dict(line=C, drug=d, split=sp, method=m,
                                  n_control_pred=int(len(ctrlALL) // 2), **pr))
        lm = lam_meta(ctrl_store[(sp, 0)], ctrl_store[(sp, 1)], list(obs_all.values()))
        mrows.append(dict(line=C, split=sp, n0=int(len(ctrlALL) // 2), d=A.n_vars, budget=K_CAL,
                          cal="|".join(cal), test="|".join(test_ok), **lm))
    if srows:
        append_csv(pd.DataFrame(srows), OUT.replace(".csv", "_sensitivity.csv"))
    if mrows:
        append_csv(pd.DataFrame(mrows), OUT.replace(".csv", "_meta.csv"))
    # scvi-tools keeps every training AnnData alive in class-level manager stores (~GBs per context)
    for store in ("_setup_adata_manager_store", "_per_instance_manager_store"):
        try:
            getattr(scgen.SCGEN, store).clear()
        except Exception:
            pass
    try:
        model._per_instance_manager_store.clear()
    except Exception:
        pass
    del model, train
    import gc; gc.collect(); torch.cuda.empty_cache()
    new = pd.DataFrame([r for r in rows if r["line"] == C])
    if len(new):
        append_csv(new, OUT)
    print("  [%d/%d] %-12s %d test drugs | %.0fs" % (li + 1, N_LINES, C, len(test), time.time() - t0), flush=True)

R = pd.read_csv(OUT)
METH = ["scGen", "scGen_plus_offset", "scGen_plus_calmean", "pooled_B", "pooled_B_scaled", "calibration_mean", "B_plus_offset"]
M = R.groupby("regime")[METH].mean()
print("\nmean PCC by control regime:"); print(M.round(4).to_string())
for reg in M.index:
    print("  %-12s ranking: %s" % (reg, " > ".join(M.loc[reg].sort_values(ascending=False).index)))
