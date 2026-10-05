"""Numerical primitives. Gene centering is always along the last axis."""

from __future__ import annotations

import numpy as np


def _array(x, *, ndim=None, name="array"):
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be nonempty and finite")
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    return a


def centre(x):
    a = _array(x)
    if a.ndim == 0 or a.shape[-1] < 2:
        raise ValueError("at least two genes required")
    return a - a.mean(axis=-1, keepdims=True)


def pcc(pred, truth):
    """Pearson correlation over genes; constant vectors return NaN, not zero."""
    p, t = centre(pred), centre(truth)
    if p.shape != t.shape:
        raise ValueError("prediction/truth shapes differ")
    den = np.linalg.norm(p, axis=-1) * np.linalg.norm(t, axis=-1)
    out = np.full(np.shape(den), np.nan, dtype=float)
    np.divide((p * t).sum(axis=-1), den, out=out, where=den > 0)
    return float(out) if out.ndim == 0 else out


def inner(pred, truth):
    p, t = centre(pred), centre(truth)
    if p.shape != t.shape:
        raise ValueError("prediction/truth shapes differ")
    out = (p * t).sum(axis=-1)
    return float(out) if np.ndim(out) == 0 else out


def split_control(n_or_labels, seed=0, equalise=True):
    """Disjoint cell or whole-unit splits; retain equal cell counts by default.

    Labels group measurement units, not treatments. Independence from treatment
    must be established by the experimental design, not inferred by this function.
    """
    rng = np.random.default_rng(seed)
    if isinstance(n_or_labels, (int, np.integer)) and not isinstance(n_or_labels, bool):
        n = int(n_or_labels)
        if n < 2:
            raise ValueError("at least two control cells required")
        p = rng.permutation(n)
        h = n // 2
        return p[:h], p[h : 2 * h if equalise else n]
    lab = np.asarray(n_or_labels)
    if lab.ndim != 1 or len(lab) < 2:
        raise ValueError("one unit label per cell required")
    if any(
        x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)) for x in lab
    ):
        raise ValueError("missing unit labels")
    units, cnt = np.unique(lab.astype(str), return_counts=True)
    if len(units) < 2:
        raise ValueError("at least two measurement units required")
    order = rng.permutation(len(units))
    groups = [[], []]
    sizes = [0, 0]
    for i in sorted(order, key=lambda i: -cnt[i]):
        j = int(sizes[1] < sizes[0])
        groups[j].append(units[i])
        sizes[j] += int(cnt[i])
    a, b = (
        rng.permutation(np.flatnonzero(np.isin(lab.astype(str), g))) for g in groups
    )
    if equalise:
        a, b = a[: min(len(a), len(b))], b[: min(len(a), len(b))]
    return a, b


def measure_a_directional(predict, control_A, control_B):
    """Signed projection coefficient used in the cross-fitted numerator identity."""
    a, b = _array(control_A, ndim=1), _array(control_B, ndim=1)
    if a.shape != b.shape:
        raise ValueError("control shapes differ")
    dc = centre(a - b)
    delta = centre(predict(a)) - centre(predict(b))
    if delta.shape != dc.shape:
        raise ValueError("predict must return one gene vector")
    den = float(dc @ dc)
    return float(-delta @ dc / den) if den > 0 else float("nan")


def measure_a(predict, control_A, control_B):
    """Nonnegative norm ratio; not the identity coefficient for general predictors."""
    a, b = _array(control_A, ndim=1), _array(control_B, ndim=1)
    if a.shape != b.shape:
        raise ValueError("control shapes differ")
    delta = centre(predict(a)) - centre(predict(b))
    dc = centre(a - b)
    if delta.shape != dc.shape:
        raise ValueError("predict must return one gene vector")
    den = np.linalg.norm(dc)
    return float(np.linalg.norm(delta) / den) if den > 0 else float("nan")


def measure_lambda(control_A, control_B, treated, *, center_signal=True):
    """Control error / cross-control response signal.

    center_signal=True uses gene-centered numerator AND denominator. False
    reproduces the manuscript's historical screening convention (uncentered
    denominator). Nonpositive signal is reported as infinity.
    """
    a, b = _array(control_A, ndim=1), _array(control_B, ndim=1)
    t = _array(treated, ndim=2)
    if a.shape != b.shape or t.shape[1:] != a.shape:
        raise ValueError("gene axes differ")
    noise = float(np.sum(centre(a - b) ** 2) / 2)
    da, db = t - a, t - b
    if center_signal:
        da, db = centre(da), centre(db)
    signal = float(np.mean(np.sum(da * db, axis=1)))
    return noise / signal if signal > 0 else float("inf")


def predicted_gap(a, control_A, control_B):
    ca, cb = _array(control_A, ndim=1), _array(control_B, ndim=1)
    if ca.shape != cb.shape:
        raise ValueError("control shapes differ")
    if not np.isfinite(a):
        raise ValueError("coefficient must be finite")
    return float(a) * float(np.sum(centre(ca - cb) ** 2) / 2)


def dual_regime_scores(predict, truth_of, control_A, control_B, targets):
    targets = list(targets)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("unique nonempty targets required")
    h = [_array(control_A, ndim=1), _array(control_B, ndim=1)]
    if h[0].shape != h[1].shape:
        raise ValueError("control shapes differ")
    out = []
    for f in (0, 1):
        preds = predict(h[f])
        for regime, c in [("shared", h[f]), ("independent", h[1 - f])]:
            for t in targets:
                y = truth_of(t, c)
                out.append(
                    dict(
                        fold=f,
                        regime=regime,
                        target=t,
                        pcc=pcc(preds[t], y),
                        inner=inner(preds[t], y),
                    )
                )
    return out


def check_identity(rows, a, control_A, control_B, tol=1e-6):
    """Require the complete two-fold, two-regime design; never drop missing rows."""
    if tol <= 0:
        raise ValueError("tol must be positive")
    acc = {}
    for row in rows:
        key = (row["target"], row["fold"], row["regime"])
        if key in acc:
            raise ValueError("duplicate scoring row")
        if row["fold"] not in (0, 1) or row["regime"] not in ("shared", "independent"):
            raise ValueError("unknown fold/regime")
        if not np.isfinite(row["inner"]):
            raise ValueError("nonfinite inner product")
        acc[key] = row["inner"]
    targets = {t for t, _, _ in acc}
    if not targets or any(
        (t, f, r) not in acc
        for t in targets
        for f in (0, 1)
        for r in ("shared", "independent")
    ):
        raise ValueError("complete paired folds required")
    obs = float(
        np.mean(
            [
                acc[t, f, "shared"] - acc[t, f, "independent"]
                for t in targets
                for f in (0, 1)
            ]
        )
    )
    pred = predicted_gap(a, control_A, control_B)
    error = abs(obs - pred) / abs(pred) if pred else abs(obs)
    return bool(error < tol), obs, pred, float(error)


def inflation_from_theory(a, lam, r2):
    """Approximate PCC inflation, not an exact sample-level PCC identity."""
    a, lam, r2 = map(float, (a, lam, r2))
    if not all(np.isfinite(v) for v in (a, lam, r2)) or lam < 0 or r2 < 0:
        raise ValueError("finite a and nonnegative lambda/r2 required")
    den = np.sqrt((r2 + a * a * lam) * (1 + lam))
    return float(a * lam / den) if den > 0 else float("nan")
