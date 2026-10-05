"""Synthetic demonstration, not manuscript evidence."""

import numpy as np
from safedelta import SplitControlAdapter

rng = np.random.default_rng(42)
base = rng.normal(size=(12, 100))
source = base[None] + rng.normal(scale=0.05, size=(4, 12, 100))
shift = rng.normal(scale=0.4, size=100)
control_a, control_b = rng.normal(scale=0.02, size=(2, 100))
cal = np.arange(5)
model = SplitControlAdapter(adapter="offset", convention="symmetric").fit(
    source, cal, base[cal] + shift, control_a, control_b
)
print(model.decision_)
print("Held-out predictions:", model.predict(np.arange(5, 12)).shape)
