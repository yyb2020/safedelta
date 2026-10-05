"""trVAE under shared vs independent controls, using the manuscript's own
architecture, hyperparameters and reconstruct() implementation.

Model/training settings copied verbatim from the authors' run_trvae_ft_all_v2.py:
  TRVAE(hidden_layer_sizes=[128,128], latent_dim=10, dr_rate=.2, recon_loss='mse', use_bn=True)
  train(n_epochs=1000, lr=1e-3, eps=.01, alpha_epoch_anneal=200, batch_size=512,
        clip_value=100, train_frac=.9, use_stratified_split=True, early_stopping ...)
  fine-tune on rehearsal(source + held_allowed): 200 epochs, lr=1e-4, train_frac=1.0
reconstruct() is imported from the capsule, not reimplemented.
"""
import sys, os, time, argparse
import numpy as np, pandas as pd, torch
import anndata as ad
if not hasattr(ad, "read"):
    ad.read = ad.read_h5ad          # scArches 0.6.1 legacy alias; authors' own note
import scarches as sca
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coupling_probe import probe, lam_meta, append_csv
sys.path.insert(0, "/projects/xunixibao/files_new/code/scripts")
from run_scperturbench_trvae_fewshot_trial_v1 import reconstruct

EARLY = {"early_stopping_metric": "val_unweighted_loss", "threshold": 0,
         "patience": 100, "reduce_lr": False}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["sciplex3", "tahoe"])
ap.add_argument("--budget", type=int, default=5)
ap.add_argument("--epochs", type=int, default=1000)
ap.add_argument("--ft-epochs", type=int, default=200)
ap.add_argument("--out", default=None)
ap.add_argument("--max-contexts", type=int, default=0)
ap.add_argument("--splits", type=int, default=3,
                help="random disjoint control partitions; each half serves as prediction side once")
A = ap.parse_args()
OUT = A.out or f"trvae_{A.dataset}_regime.csv"
rng = np.random.default_rng(0)


def load_sciplex3():
    a = ad.read_h5ad("/projects/xunixibao/files_new/raw/datasets/sciplex3.h5ad")
    a.obs["condition1"] = a.obs.condition1.astype(str)
    a.obs["condition2"] = a.obs.condition2.astype(str)
    return a


def load_tahoe():
    Z = np.load("plate7_cells_subset.npz", allow_pickle=True)
    X = Z["X"]; grp = Z["group"]
    line = np.array([str(x) for x in Z["line"]])[grp]
    drug = np.array([str(x) for x in Z["drug"]])[grp]
    drug = np.where(drug == "CONTROL", "control", drug)
    a = ad.AnnData(X.astype(np.float32),
                   obs=pd.DataFrame({"condition1": line, "condition2": drug},
                                    index=[f"c{i}" for i in range(len(line))]),
                   var=pd.DataFrame(index=[str(x) for x in Z["genes"]]))
    import scanpy as sc
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    return a


A_ = load_sciplex3() if A.dataset == "sciplex3" else load_tahoe()
CTX = sorted(A_.obs.condition1.unique())
DRUGS = sorted(d for d in A_.obs.condition2.unique() if d != "control")
print("%s: %d cells x %d genes | %d contexts | %d drugs"
      % (A.dataset, A_.n_obs, A_.n_vars, len(CTX), len(DRUGS)), flush=True)

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
    v = np.asarray(v, dtype=np.float64); v = v - v.mean()
    return v / max(np.linalg.norm(v), 1e-12)


done = set()
if os.path.exists(OUT):
    done = set(pd.read_csv(OUT).context.unique())
    print("resuming; done: %s" % sorted(done), flush=True)

K = min(A.budget, max(1, len(DRUGS) - 2))
if A.max_contexts:
    CTX = CTX[:A.max_contexts]
for C in CTX:
    if C in done:
        continue
    t0 = time.time()
    cal = list(rng.choice(DRUGS, K, replace=False))
    test = [d for d in DRUGS if d not in cal]
    o1 = A_.obs.condition1.values; o2 = A_.obs.condition2.values
    source = A_[o1 != C].copy()
    held_allowed = A_[(o1 == C) & np.isin(o2, cal + ["control"])].copy()
    conditions = sorted(source.obs.condition2.unique())
    if any(d not in conditions for d in test):
        test = [d for d in test if d in conditions]
    if not test:
        continue
    source.X = np.asarray(source.X, dtype=np.float32)

    model = sca.models.TRVAE(source, condition_key="condition2", conditions=conditions,
                             hidden_layer_sizes=[128, 128], latent_dim=10, dr_rate=.2,
                             recon_loss="mse", use_bn=True)
    model.train(n_epochs=A.epochs, lr=1e-3, eps=.01, alpha_epoch_anneal=200, batch_size=512,
                clip_value=100, train_frac=.9, use_stratified_split=True, seed=0,
                monitor=False, early_stopping_kwargs=EARLY)
    ck = f"/tmp/trvae_ck_{A.dataset}_{abs(hash(C)) % 99999}"
    model.save(ck, overwrite=True)
    n = max(source.n_obs, held_allowed.n_obs)
    src = source[rng.choice(source.n_obs, n, replace=source.n_obs < n)].copy()
    tgt = held_allowed[rng.choice(held_allowed.n_obs, n, replace=held_allowed.n_obs < n)].copy()
    rehearsal = ad.AnnData(np.concatenate([np.asarray(src.X, np.float32), np.asarray(tgt.X, np.float32)]),
                           obs=pd.concat([src.obs.copy(), tgt.obs.copy()]), var=source.var.copy())
    adapted = sca.models.TRVAE.load(ck, adata=rehearsal, map_location="cpu")
    adapted.train(n_epochs=A.ft_epochs, lr=1e-4, eps=.01, batch_size=512, clip_value=100,
                  train_frac=1.0, use_early_stopping=False, seed=0, monitor=False)

    other = o1 != C
    pooled = {d: np.asarray(A_.X[other & (o2 == d)]).mean(0, dtype=np.float64)
                 - np.asarray(A_.X[other & (o2 == "control")]).mean(0, dtype=np.float64) for d in test}
    obs_expr = {d: np.asarray(A_.X[(o1 == C) & (o2 == d)]).mean(0, dtype=np.float64) for d in test}
    # all drugs with cells in this context (for lambda) and the calibration drugs (for the residual offset)
    obs_all = {d: np.asarray(A_.X[(o1 == C) & (o2 == d)]).mean(0, dtype=np.float64)
               for d in DRUGS if ((o1 == C) & (o2 == d)).sum() >= 5}
    ctrl_idx = np.where((o1 == C) & (o2 == "control"))[0]

    # EQUAL-MARGINAL-DEPTH, CROSS-FITTED DESIGN.
    # Both regimes use an n/2 control estimate for the prediction AND an n/2 control estimate
    # for the truth, so marginal control precision is identical. The prediction is literally the
    # same array in both regimes; the ONLY difference is whether the truth's control is the same
    # realization as the prediction's (shared) or an independent one (independent). This isolates
    # shared-error coupling from control-estimate precision — the earlier cALL-vs-cA/cB design
    # confounded the two, because its independent arm halved the control depth.
    # Each half serves as the prediction side once (cross-fitting), repeated over A.splits
    # random disjoint partitions. Model training does not depend on the split, so this is cheap.
    rows = []
    sens_store = {}          # (split, fold) -> {method: prediction vector}, for the sensitivity probe
    ctrl_store = {}          # (split, fold) -> control estimate used on the prediction side
    for s in range(A.splits):
        rs = np.random.default_rng(1000 + s)
        perm = rs.permutation(len(ctrl_idx))
        halves = [ctrl_idx[perm[:len(perm) // 2]], ctrl_idx[perm[len(perm) // 2:]]]
        for fold in (0, 1):
            ip, it = halves[fold], halves[1 - fold]
            cP = np.asarray(A_.X[ip]).mean(0, dtype=np.float64)
            cT = np.asarray(A_.X[it]).mean(0, dtype=np.float64)
            basis = A_[ip].copy()
            cal_mean = np.mean([np.asarray(A_.X[(o1 == C) & (o2 == d)]).mean(0, dtype=np.float64) - cP
                                for d in cal], 0)
            # scalar magnitude calibration fitted on calibration perturbations only
            pooled_cal = {d: np.asarray(A_.X[other & (o2 == d)]).mean(0, dtype=np.float64)
                             - np.asarray(A_.X[other & (o2 == "control")]).mean(0, dtype=np.float64)
                          for d in cal}
            cal_obs = [np.asarray(A_.X[(o1 == C) & (o2 == d)]).mean(0, dtype=np.float64) - cP for d in cal]
            s_hat = _fit_scalar([pooled_cal[d] for d in cal], cal_obs)
            # residual-form context offset: mean over calibration drugs of [(T_cal - cP) - (rec_cal - cP)]
            rec_cal = {d: reconstruct(adapted, basis, d) for d in cal}
            off_V = np.mean([co - (rec_cal[d] - cP) for d, co in zip(cal, cal_obs)], 0)
            for d in test:
                rec = reconstruct(adapted, basis, d)      # authors' own function
                preds = {"trVAE": rec - cP,
                         "trVAE_plus_offset": rec - cP + off_V,        # residual form, a ~ 1
                         "trVAE_plus_calmean": rec - cP + cal_mean,    # additive form (old name trVAE_plus_offset), a ~ a_trVAE + 1
                         "pooled_B": pooled[d],
                         "pooled_B_scaled": s_hat * pooled[d],
                         "calibration_mean": cal_mean,
                         "B_plus_offset": pooled[d] + cal_mean}
                for regime, ctrue in (("shared", cP), ("independent", cT)):
                    truth = obs_expr[d] - ctrue
                    row = dict(dataset=A.dataset, context=C, drug=d, regime=regime,
                               split=s, fold=fold, n_control_pred=len(ip), n_control_truth=len(it),
                               s_hat=float(s_hat))
                    for nm, pv in preds.items():
                        pcc, r2, dot = _scores(pv, truth)
                        row[nm] = pcc; row[nm + "__r2"] = r2; row[nm + "__dot"] = dot
                    rows.append(row)
                sens_store.setdefault((s, fold), {})[d] = {k_: np.asarray(v_, float) for k_, v_ in preds.items()}
            ctrl_store[(s, fold)] = cP
    # SENSITIVITY PROBE (free): fold 0 and fold 1 of the same split use two different control
    # realizations, so the prediction has already been computed twice under a perturbed control.
    # a_hat = ||P(fold0) - P(fold1)|| / ||c(fold0) - c(fold1)|| measures how much of the control
    # error the method carries into its prediction. Uses controls + calibration only — never the
    # held-out truth — so it is computable before any evaluation.
    srows, mrows = [], []
    for sp in range(A.splits):
        if (sp, 0) not in ctrl_store or (sp, 1) not in ctrl_store:
            continue
        common = set(sens_store.get((sp, 0), {})) & set(sens_store.get((sp, 1), {}))
        for d in common:
            for m in sens_store[(sp, 0)][d]:
                pr = probe(sens_store[(sp, 0)][d][m], sens_store[(sp, 1)][d][m],
                           ctrl_store[(sp, 0)], ctrl_store[(sp, 1)])
                srows.append(dict(dataset=A.dataset, context=C, drug=d, split=sp, method=m,
                                  n_control_pred=int(len(ctrl_idx) // 2), **pr))
        lm = lam_meta(ctrl_store[(sp, 0)], ctrl_store[(sp, 1)], list(obs_all.values()))
        mrows.append(dict(dataset=A.dataset, context=C, split=sp, n0=int(len(ctrl_idx) // 2),
                          d=A_.n_vars, budget=K, cal="|".join(cal), test="|".join(test), **lm))
    if srows:
        append_csv(pd.DataFrame(srows), OUT.replace(".csv", "_sensitivity.csv"))
    if mrows:
        append_csv(pd.DataFrame(mrows), OUT.replace(".csv", "_meta.csv"))
    append_csv(pd.DataFrame(rows), OUT)
    print("  %-10s %d test drugs | %.0fs" % (C, len(test), time.time() - t0), flush=True)

R = pd.read_csv(OUT)
METH = ["trVAE", "trVAE_plus_offset", "trVAE_plus_calmean", "pooled_B", "pooled_B_scaled", "calibration_mean", "B_plus_offset"]
METH = [m for m in METH if m in R.columns]
M = R.groupby("regime")[METH].mean()
print("\n%s — mean PCC by control regime:" % A.dataset); print(M.round(4).to_string())
for reg in M.index:
    print("  %-12s %s" % (reg, " > ".join(M.loc[reg].sort_values(ascending=False).index)))
