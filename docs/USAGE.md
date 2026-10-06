# Usage

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

## Development

```bash
python -m pip install .
python -m pip install '.[test]' && python -m pytest
```
