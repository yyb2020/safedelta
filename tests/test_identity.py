"""The cross-fitted identity is algebraic, so these are exact-value tests."""

import numpy as np

import safedelta as cc

D, N0, K = 400, 40, 5
rng = np.random.default_rng(0)


def _world(a):
    """Signal, two independent equal-depth control estimates, and a predictor whose
    control carriage coefficient is exactly `a`."""
    sig = rng.normal(0, 0.05, D)
    cA = rng.normal(0, 1 / np.sqrt(N0), D)
    cB = rng.normal(0, 1 / np.sqrt(N0), D)
    P0 = rng.normal(0, 0.05, D)
    predict = lambda c: {0: P0 - a * c}
    truth_of = lambda t, c: sig - c
    return sig, cA, cB, predict, truth_of


def test_measure_a_recovers_the_coefficient():
    for a in (0.0, 1.0, 2.0, 0.37):
        _, cA, cB, predict, _ = _world(a)
        got = cc.measure_a(lambda c: predict(c)[0], cA, cB)
        assert abs(got - a) < 1e-9, (a, got)


def test_gap_identity_is_exact_under_cross_fitting():
    for a in (0.5, 1.0, 2.0):
        _, cA, cB, predict, truth_of = _world(a)
        rows = cc.dual_regime_scores(predict, truth_of, cA, cB, [0])
        ok, obs, pred, rel = cc.check_identity(rows, a, cA, cB)
        assert ok, (a, obs, pred, rel)


def test_a_zero_null_is_exact_not_merely_unbiased():
    _, cA, cB, predict, truth_of = _world(0.0)
    rows = cc.dual_regime_scores(predict, truth_of, cA, cB, [0])
    fold_avg = {}
    for r in rows:
        fold_avg.setdefault(r["regime"], []).append(r["inner"])
    gap = np.mean(fold_avg["shared"]) - np.mean(fold_avg["independent"])
    assert abs(gap) < 1e-9, gap


def test_uncentred_norm_is_the_classic_mistake():
    """Comparing the centred gap against a RAW control norm inflates the denominator
    whenever the two halves differ in overall mean, so the ratio drifts off 1."""
    _, cA, cB, predict, truth_of = _world(1.0)
    cB = cB + 0.02  # a systematic depth offset
    rows = cc.dual_regime_scores(predict, truth_of, cA, cB, [0])
    ok, obs, pred, _ = cc.check_identity(rows, 1.0, cA, cB)
    raw = float(np.sum((cA - cB) ** 2) / 2)
    assert ok, "centred comparison must still pass"
    assert abs(obs - raw) / raw > 1e-3, "raw norm should NOT match the centred gap"


def test_split_control_equalises_depth_and_refuses_single_unit():
    lab = np.array(["p1"] * 30 + ["p2"] * 70)
    ia, ib = cc.split_control(lab, seed=1)
    assert len(ia) == len(ib) == 30
    assert not set(ia) & set(ib)
    try:
        cc.split_control(np.array(["only"] * 50))
    except ValueError:
        pass
    else:
        raise AssertionError("a single measurement unit must raise")


def test_inflation_is_not_a_function_of_a_alone():
    """Same `a`, different relative norm -> several-fold different PCC inflation.
    This is why the unnormalised gap is the statistic to report."""
    hi = cc.inflation_from_theory(a=0.38, lam=0.15, r2=0.05)
    lo = cc.inflation_from_theory(a=0.38, lam=0.15, r2=1.00)
    assert hi / lo > 2.0, (hi, lo)


def test_directional_a_is_exact_for_anisotropic_carriage():
    """A projected (non-scalar) carriage breaks the norm-ratio estimator but not the
    directional one.  This is the case a real low-rank component produces."""
    sig = rng.normal(0, 0.05, D)
    cA, cB = (rng.normal(0, 1 / np.sqrt(N0), D) for _ in range(2))
    P0 = rng.normal(0, 0.05, D)
    Q = np.linalg.svd(rng.normal(size=(16, D)), full_matrices=False)[
        2
    ].T  # rank-16 basis
    predict = lambda c: {0: P0 - 0.5 * c - 0.5 * (Q @ (Q.T @ c))}  # anisotropic
    truth_of = lambda t, c: sig - c
    rows = cc.dual_regime_scores(predict, truth_of, cA, cB, [0])
    a_norm = cc.measure_a(lambda c: predict(c)[0], cA, cB)
    a_dir = cc.measure_a_directional(lambda c: predict(c)[0], cA, cB)
    ok_norm, _, _, rel_norm = cc.check_identity(rows, a_norm, cA, cB)
    ok_dir, _, _, rel_dir = cc.check_identity(rows, a_dir, cA, cB)
    assert ok_dir, ("directional must be exact", rel_dir)
    assert not ok_norm and rel_norm > 1e-4, ("norm ratio should be off here", rel_norm)


def test_the_two_estimators_agree_for_scalar_carriage():
    _, cA, cB, predict, _ = _world(1.7)
    f = lambda c: predict(c)[0]
    assert abs(cc.measure_a(f, cA, cB) - cc.measure_a_directional(f, cA, cB)) < 1e-9


def test_negative_directional_coefficient_preserves_the_sign_of_the_gap():
    _, control_a, control_b, predict, truth = _world(-0.7)
    directional = cc.measure_a_directional(
        lambda control: predict(control)[0], control_a, control_b
    )
    norm_ratio = cc.measure_a(lambda control: predict(control)[0], control_a, control_b)
    assert directional < 0 < norm_ratio
    rows = cc.dual_regime_scores(predict, truth, control_a, control_b, [0])
    ok, observed, expected, _ = cc.check_identity(
        rows, directional, control_a, control_b
    )
    assert ok and observed < 0 and expected < 0
