# SafeDelta

A split-control criterion and diagnostics for few-shot adaptation of
perturbation-response predictors. NumPy is the only runtime dependency.
Python 3.10 or newer. Version 1.0.0.

Few-shot adaptation corrects a shared predictor with a handful of responses measured
in a new cellular context. Whether that correction helps or harms is not known before
the remaining responses are measured, and the usual evidence is inflated because every
response in a context is referenced to one control estimate. SafeDelta splits the
controls of the target context into two disjoint halves, estimates the correction
against one half and evaluates it against the other, and releases the correction only
when the resulting leave-one-out evidence is positive. Otherwise it returns the shared
predictor unchanged.

The package implements

- the criterion and the release decision (`SplitControlAdapter`, `ReleaseDecision`);
- two adapters, `offset` and `mixture`;
- estimators of the carriage coefficient (`measure_a_directional`, `measure_a`) and of
  the control noise-to-signal ratio (`measure_lambda`);
- the paired shared/independent scoring and the exact inner-product identity
  (`dual_regime_scores`, `check_identity`).

Source code: [GitHub](https://github.com/yyb2020/safedelta).
Package: [PyPI](https://pypi.org/project/safedelta/).

## Installation

```bash
python -m pip install safedelta
python -m pip install .                    # from a checkout
python -m pip install '.[test]' && python -m pytest
```

## Minimal example

```python
from safedelta import SplitControlAdapter

model = SplitControlAdapter(adapter="offset", convention="symmetric").fit(
    source_responses, calibration_indices, treated_calibration, control_a, control_b
)
print(model.decision_)          # evidence, released, convention, adapter, ...
prediction = model.predict(test_indices)
```

`python examples/quickstart.py` runs the same call on simulated arrays.

### Inputs

Arrays must share an explicitly aligned gene axis.

| Argument | Shape | Content |
|---|---|---|
| `source_responses` | (source contexts, all targets, genes) | response deltas of the other contexts |
| `calibration_indices` | (calibration targets,) | position of each calibration row on the target axis |
| `treated_calibration` | (calibration targets, genes) | expression of the calibration targets **before control subtraction** |
| `control_a`, `control_b` | (genes,) | mean profiles of two disjoint control halves |

Neither held-out target expression nor held-out outcomes enter `fit`.

### Adapters and conventions

The `offset` adapter estimates the average calibration residual. `mixture` fits a
simplex mixture of the pooled source, residual-offset, source-weighted and
source-basis-projected candidates. `one_way` evaluates A-fitted calibration
predictions against B controls; `symmetric` averages both directions. Positive
leave-one-calibration-target-out PCC gain releases the adapter. Otherwise `predict`
returns the pooled source exactly. This is an empirical decision, not a guarantee of
benefit for every target. `predict(fold=0)` and `fold=1` expose the two fitted
candidates under the same decision.

### Preparing data

For raw counts use `log_normalize_counts` on the **full gene library before gene
selection**, then `pseudobulk_mean`. Already processed public expression matrices are
used on their publisher-provided scale; do not normalise them twice. Independent
control units must come from the experimental design. `split_control(unit_labels)`
keeps measurement units separate and equalises cell counts by default. It cannot
establish biological independence itself.

### Command line

The command-line interface reads an NPZ with the five named input arrays:

```bash
safedelta fit input.npz --output result --adapter offset --convention symmetric
```

Outputs are predictions for both folds and a JSON release decision. Pickled arrays are
not accepted. Constant-vector PCC is undefined and leads to abstention.

## Diagnostics

`measure_a_directional` estimates the signed projection coefficient used in the exact
centred-inner-product identity. `measure_a` is a non-negative norm ratio; the two are
not interchangeable for projected or negative carriage. `dual_regime_scores` and
`check_identity` require paired targets and both folds. `inflation_from_theory` is an
approximation for PCC, not an exact PCC identity. `measure_lambda` centres signal and
noise by default; `center_signal=False` explicitly requests the historical
uncentred-signal screening convention.

## Analysis code and key results

The optional `paper/` directory holds the workflows that recompute the main analyses
from public matrices, together with the key result tables. It is not part of the
Python wheel.

| Directory | Content |
|---|---|
| `paper/workflows/`, `paper/reproduce.py`, `paper/verify.py` | workflows that start from public expression or count matrices: `external`, `sciplex`, `tahoe`, `liver`, `weights` |
| `paper/verified_tables/` | 20 tables recomputed by those workflows |
| `paper/reference_tables/` | 36 frozen result tables: the 20 that the workflows are compared against and 16 summary tables of the other main results |
| `paper/legacy_sources/` | scripts used to run the published models (trVAE, scGen, scPRAM) and the weight comparisons; kept for provenance, only three of them are re-run by the `weights` workflow |
| `paper/specs/`, `paper/manifests/`, `paper/audit/` | cohort specifications, file hashes and the comparison receipts of the audited runs |

### Key result tables

| Result | Table in `paper/reference_tables/` |
|---|---|
| Criterion against held-out gain on the atlas plates | `criterion_predicts_transfer.csv`, `release_rule_validation.csv`, `criterion_threshold_loco_v2.csv` |
| Criterion on four external datasets | `criterion_external_summary.csv`, `criterion_external_panels.csv`, `fig1d_external_validation.csv` |
| Thirteen prediction configurations under shared and independent controls | `w10_four_dataset_verified.csv`, `iid_arm_ranking.csv`, `w10_reproduction_paired.csv` |
| Adjacent-rank margins of a published leaderboard and the depth projection | `nm_leaderboard_identifiability.csv`, `nm_identifiability_shared_regime.csv`, `nm_bias_vs_margin.csv`, `nm_design_requirement.csv` |
| Carriage coefficient by model and dataset, and its additivity | `fig3b_carriage_by_model_dataset.csv`, `carriage_additivity.csv`, `composition_rule_carriage.csv` |
| Inflation against the noise-to-signal ratio and control depth | `lambda_screen_all_datasets.csv`, `lambda_not_depth.csv`, `plate7_depth_law_summary.csv`, `sciplex3_depth_lambda.csv`, `sciplex3_depth_numerator_gap.csv` |
| Adapter capacity and weight rules | `plate7_depth_ladder.csv`, `weightrule_summary.csv`, `weightrule2_summary.csv`, `lib_ablate_summary.csv` |
| Liver fibrosis programme | `liver_core94_signature.csv`, `liver_spatial_section_contrast.csv`, `liver_core_vs_specific_per_section.csv`, `liver_cohort_stage_association.csv`, `liver_core_external_validation.csv` |

### Recomputing from public matrices

```bash
python -m pip install '.[reproduce]'
python scripts/check_integrity.py                 # hashes of the stored tables
python scripts/download_public_matrices.py --output /data/public_matrices
python paper/reproduce.py --config paths.json --workflow external --output paper/generated/run01
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the configuration file, the
scope of each workflow and the comparisons it performs.

### Coverage, stated plainly

- The workflows recompute 20 tables from public matrices and compare them with the
  stored versions: the external-dataset criterion on the pooled predictor, the
  sciPlex3 depth experiment, the atlas-plate criterion and adapter ladder, the weight
  comparisons and the liver analysis. Sixteen comparisons cover complete tables;
  four cover declared subsets: the pooled-predictor rows of the two external
  criterion tables and selected columns of the liver signature and spot-score tables.
- `--workflow all` means all *implemented* workflows, not every experiment in the
  article. In the complete analysis there are 79 plot tables; 15 are fully recomputed
  here and 3 partially. `paper/manifests/table_lineage.tsv` lists the status of each.
- Predictions of third-party models (trVAE, scGen, scPRAM) require separately trained
  models. The scripts that produced them are in `paper/legacy_sources/` but are not
  re-run by the workflows, and their outputs are stored as summary tables only.
- Figure-rendering code, per-panel plot tables, raw data, fitted checkpoints and the
  manuscript are not distributed here.

## Citation

See [CITATION.cff](https://github.com/yyb2020/safedelta/blob/main/CITATION.cff)
for the software citation.

## Licence

MIT, see [LICENSE](LICENSE).
