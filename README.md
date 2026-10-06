# SafeDelta

A split-control criterion for few-shot adaptation in single-cell perturbation
prediction. SafeDelta uses calibration evidence to decide whether to adapt a
shared predictor; otherwise, it retains the shared prediction.

Version 1.0.0 · Python ≥3.10 · NumPy

## Installation

```bash
pip install safedelta
```

## Quick start

```python
from safedelta import SplitControlAdapter

model = SplitControlAdapter(adapter="offset", convention="symmetric").fit(
    source_responses, calibration_indices, treated_calibration, control_a, control_b
)
prediction = model.predict(test_indices)
print(model.decision_)
```

See [usage](docs/USAGE.md) for input shapes, adapters and the command line.
Run [examples/quickstart.py](examples/quickstart.py) for a complete simulated example.

## Reproducibility

Analysis workflows and result tables are in [`paper/`](paper/).
See the [reproduction guide](docs/REPRODUCIBILITY.md) for data, commands and coverage.

## Citation and licence

[Citation](CITATION.cff) · [MIT licence](LICENSE) · [PyPI](https://pypi.org/project/safedelta/)
