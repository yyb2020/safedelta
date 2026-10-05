"""Independent archived numerical definitions protect the scientific contract."""

import importlib.util
from pathlib import Path

import numpy as np
from safedelta.adapter import _candidate, source_basis

from safedelta import SplitControlAdapter

p = Path(__file__).resolve().parents[1] / "paper/workflows/historical_math.py"
spec = importlib.util.spec_from_file_location("archived_math", p)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def test_mixture_matches_historical_predictions_weights_and_release():
    rng = np.random.default_rng(2026)
    source = rng.normal(size=(4, 9, 30))
    cal = np.array([1, 2, 5, 7, 8])
    treated = rng.normal(size=(5, 30))
    ca, cb = rng.normal(size=(2, 30))
    basis = old.sbasis(source)
    expected = old.fit_pair2(
        source.mean(0, keepdims=True), source, treated - ca, cal, basis
    )
    got = _candidate(source, treated - ca, cal, source_basis(source), "mixture")
    for a, b in zip(expected[:3], got[:3]):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
    m = SplitControlAdapter(adapter="mixture", convention="one_way").fit(
        source, cal, treated, ca, cb
    )
    values = []
    for j, t in enumerate(cal):
        keep = np.arange(5) != j
        b, c, _, _ = old.fit_pair2(
            source.mean(0, keepdims=True),
            source,
            (treated - ca)[keep],
            cal[keep],
            basis,
        )
        values.append(old.rowp(c[t], treated[j] - cb) - old.rowp(b[t], treated[j] - cb))
    np.testing.assert_allclose(m.decision_.evidence, np.mean(values), atol=1e-12)
    np.testing.assert_allclose(
        m.predict(), expected[1] if np.mean(values) > 0 else expected[0], atol=1e-12
    )
