import json

import numpy as np
import pytest
from safedelta.cli import main

from safedelta import (
    SplitControlAdapter,
    centre,
    check_identity,
    log_normalize_counts,
    measure_lambda,
    pcc,
    pseudobulk_mean,
)


def inputs(offset=1.0):
    r = np.random.default_rng(18)
    base = r.normal(size=(8, 24))
    shift = offset * r.normal(size=24)
    return dict(
        source_responses=np.stack([base, base]),
        calibration_indices=np.array([0, 1, 2, 3]),
        treated_calibration=base[:4] + shift,
        control_a=np.zeros(24),
        control_b=np.zeros(24),
    )


@pytest.mark.parametrize("kind", ["offset", "mixture"])
def test_release_improves_unseen_targets(kind):
    data = inputs()
    m = SplitControlAdapter(adapter=kind).fit(**data)
    assert m.decision_.released
    truth = data["source_responses"][0, 4:] + (
        data["treated_calibration"][0] - data["source_responses"][0, 0]
    )
    assert np.mean(pcc(m.predict([4, 5, 6, 7]), truth)) > np.mean(
        pcc(m.base_[0, 4:], truth)
    )
    assert np.allclose(m.weights_.sum(axis=1), 1)


def test_abstain_returns_exact_base_and_copy():
    d = inputs(0.0)
    m = SplitControlAdapter().fit(**d)
    assert not m.decision_.released
    assert np.array_equal(m.predict(), d["source_responses"].mean(0))
    prediction = m.predict()
    prediction[:] = -100
    assert np.array_equal(m.predict(), m.base_[0])


def test_split_control_evidence_penalizes_shared_control_noise():
    d = inputs(0.0)
    d["control_a"] = np.random.default_rng(2).normal(size=24) * 3
    m = SplitControlAdapter(convention="one_way").fit(**d)
    assert m.decision_.evidence < 0
    assert not m.decision_.released


def test_failed_refit_is_not_predictable():
    m = SplitControlAdapter().fit(**inputs())
    d = inputs()
    d["calibration_indices"] = np.array([0, 0, 2, 3])
    with pytest.raises(ValueError):
        m.fit(**d)
    with pytest.raises(RuntimeError):
        m.predict()


def test_last_axis_centering_and_constants():
    assert np.allclose(centre([[1, 2, 3], [10, 20, 30]]).mean(-1), 0)
    assert np.isnan(pcc([1, 1], [1, 2]))
    with pytest.raises(ValueError):
        pcc([1, np.nan], [1, 2])
    with pytest.raises(ValueError):
        check_identity(
            [dict(target=0, fold=0, regime="shared", inner=1)], 1, [1, 2], [2, 1]
        )


def test_count_preprocessing_order_and_all_gene_library():
    x = np.array([[1.0, 1.0, 8.0], [9.0, 1.0, 0.0]])
    y = log_normalize_counts(x, target_sum=10)
    assert np.allclose(y, np.log1p(x))
    names, means = pseudobulk_mean(y, ["same", "same"])
    assert names.tolist() == ["same"]
    assert np.allclose(means[0], np.log1p(x).mean(0))
    assert not np.allclose(means[0], np.log1p(x.mean(0)))
    with pytest.raises(ValueError):
        log_normalize_counts([[0, 0]])


def test_centered_lambda_is_invariant_to_global_treated_shift():
    c1 = np.array([1.0, 0.0, -1.0])
    c2 = -c1
    t = np.array([[8.0, 1.0, -5.0], [4.0, 5.0, -6.0]])
    assert measure_lambda(c1, c2, t) == pytest.approx(measure_lambda(c1, c2, t + 100))
    assert measure_lambda(c1, c2, t, center_signal=False) != pytest.approx(
        measure_lambda(c1, c2, t + 100, center_signal=False)
    )


def test_cli_roundtrip_and_failure(tmp_path):
    inp = tmp_path / "input.npz"
    np.savez(inp, **inputs())
    assert main(["fit", str(inp), "--output", str(tmp_path / "out")]) == 0
    result = json.loads((tmp_path / "out/decision.json").read_text())
    assert result["released"]
    with np.load(tmp_path / "out/predictions.npz") as f:
        assert f["fold_a"].shape == (8, 24)
    with pytest.raises(SystemExit) as exc:
        main(["fit", str(tmp_path / "missing"), "--output", str(tmp_path / "out")])
    assert exc.value.code == 2
