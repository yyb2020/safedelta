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



### Command line

The command-line interface reads an NPZ with the five named input arrays:

```bash
safedelta fit input.npz --output result --adapter offset --convention symmetric
```

Outputs are predictions for both folds and a JSON release decision. Pickled arrays are
not accepted. Constant-vector PCC is undefined and leads to abstention.

### Recomputing from public matrices

```bash
python -m pip install '.[reproduce]'
python scripts/check_integrity.py                 # hashes of the stored tables
python scripts/download_public_matrices.py --output /data/public_matrices
python paper/reproduce.py --config paths.json --workflow external --output paper/generated/run01
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the configuration file, the
scope of each workflow and the comparisons it performs.


## Citation

See [CITATION.cff](https://github.com/yyb2020/safedelta/blob/main/CITATION.cff)
for the software citation.

## Licence

MIT, see [LICENSE](LICENSE).
