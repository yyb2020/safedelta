"""Calibration-only release decisions. No held-out outcomes are accepted by fit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import _array, pcc


def project_simplex(v):
    u = np.sort(v)[::-1]
    c = np.cumsum(u) - 1
    rho = np.flatnonzero(u - c / np.arange(1, len(u) + 1) > 0)[-1]
    return np.maximum(v - c[rho] / (rho + 1), 0)


def fit_simplex(source, truth, *, ridge=0.01, iterations=3000, tolerance=1e-12):
    """Ridge-regularized simplex fit used in the archived four-candidate adapter."""
    source = _array(source, ndim=3)
    truth = _array(truth, ndim=2)
    if source.shape[1:] != truth.shape:
        raise ValueError("source/truth shapes differ")
    n = source.shape[0]
    uniform = np.full(n, 1 / n)
    design = source.reshape(n, -1).T
    out = truth.ravel()
    gram = design.T @ design / len(out)
    linear = design.T @ out / len(out)
    lip = 2 * (float(np.linalg.eigvalsh(gram).max()) + ridge)
    w = uniform.copy()
    for _ in range(iterations):
        nw = project_simplex(
            w - 2 * (gram @ w - linear + ridge * (w - uniform)) / max(lip, 1e-12)
        )
        if np.linalg.norm(nw - w) <= tolerance:
            return nw
        w = nw
    return w


def source_basis(source, rank=16):
    s = _array(source, ndim=3)
    m = (s - s.mean(0, keepdims=True)).reshape(-1, s.shape[-1])
    _, sv, vt = np.linalg.svd(m, full_matrices=False)
    nz = int((sv > max(float(sv[0]) * 1e-10, 1e-12)).sum())
    return vt[: min(rank, nz)]


def _candidate(source, truth, cal, basis, kind):
    base = source.mean(axis=0)
    residual = (truth - base[cal]).mean(axis=0)
    if kind == "offset":
        return base, base + residual, np.array([0.0, 1.0, 0.0, 0.0]), 0
    sw = fit_simplex(source[:, cal], truth) if len(source) > 1 else np.ones(1)
    sc = np.tensordot(sw, source, axes=(0, 0))
    candidates = []
    for rank in (4, 8, 16):
        b = basis[: min(rank, len(basis))]
        pred = base + (residual @ b.T) @ b
        candidates.append((float(np.mean((pred[cal] - truth) ** 2)), rank, pred))
    _, rank, projected = min(candidates, key=lambda x: x[0])
    bank = np.stack([base, base + residual, sc, projected])
    cw = fit_simplex(bank[:, cal], truth)
    return base, np.tensordot(cw, bank, axes=(0, 0)), cw, rank


@dataclass(frozen=True)
class ReleaseDecision:
    evidence: float
    released: bool
    convention: str
    adapter: str
    calibration_size: int
    reason: str


class SplitControlAdapter:
    """Fit on calibration expression and release iff held-out-calibration PCC gain > 0.

    source_responses: (source contexts, all targets, genes), in response/delta space.
    treated_calibration: (k, genes), pseudobulk expression BEFORE subtracting controls.
    control_a/control_b: disjoint control means over the same gene axis.
    Calibration indices identify rows in the all-target axis, in matching order.

    'one_way' reproduces the atlas decision convention; 'symmetric' averages A→B
    and B→A calibration evidence, as in the external additive-offset experiments.
    Both candidate folds remain available through predict(fold=...).
    """

    def __init__(self, *, adapter="offset", convention="symmetric"):
        if adapter not in ("offset", "mixture"):
            raise ValueError("adapter must be offset or mixture")
        if convention not in ("one_way", "symmetric"):
            raise ValueError("unknown control convention")
        self.adapter = adapter
        self.convention = convention

    def fit(
        self,
        source_responses,
        calibration_indices,
        treated_calibration,
        control_a,
        control_b,
    ):
        self.__dict__.pop("decision_", None)
        source = _array(source_responses, ndim=3, name="source_responses").copy()
        treated = _array(treated_calibration, ndim=2, name="treated_calibration")
        ca, cb = _array(control_a, ndim=1), _array(control_b, ndim=1)
        cal = np.asarray(calibration_indices)
        if cal.ndim != 1 or not np.issubdtype(cal.dtype, np.integer):
            raise ValueError("integer calibration indices required")
        if len(cal) < 2 or len(np.unique(cal)) != len(cal):
            raise ValueError("at least two unique calibration targets required")
        if np.any(cal < 0) or np.any(cal >= source.shape[1]):
            raise ValueError("calibration index outside source target axis")
        if (
            treated.shape != (len(cal), source.shape[2])
            or ca.shape != cb.shape
            or ca.shape != (source.shape[2],)
        ):
            raise ValueError("target/gene axes do not match")
        if source.shape[2] < 2:
            raise ValueError("at least two genes required")
        basis = (
            source_basis(source)
            if self.adapter == "mixture"
            else np.empty((0, source.shape[-1]))
        )
        h = [ca, cb]
        evidence = []
        bases = []
        candidates = []
        weights = []
        for fold in (0, 1):
            y = treated - h[fold]
            base, cand, cw, rank = _candidate(source, y, cal, basis, self.adapter)
            bases.append(base)
            candidates.append(cand)
            weights.append(cw)
            if self.convention == "one_way" and fold == 1:
                continue
            for j, target in enumerate(cal):
                keep = np.arange(len(cal)) != j
                b, p, _, _ = _candidate(source, y[keep], cal[keep], basis, self.adapter)
                truth = treated[j] - h[1 - fold]
                evidence.append(pcc(p[target], truth) - pcc(b[target], truth))
        value = float(np.mean(evidence))
        release = bool(np.isfinite(value) and value > 0)
        reason = (
            "positive_calibration_evidence"
            if release
            else (
                "nonfinite_calibration_evidence"
                if not np.isfinite(value)
                else "nonpositive_calibration_evidence"
            )
        )
        self.base_ = np.stack(bases)
        self.candidate_ = np.stack(candidates)
        self.weights_ = np.stack(weights)
        for a in [self.base_, self.candidate_, self.weights_]:
            a.flags.writeable = False
        self.decision_ = ReleaseDecision(
            value, release, self.convention, self.adapter, len(cal), reason
        )
        return self

    def predict(self, indices=None, *, fold=0):
        if not hasattr(self, "decision_"):
            raise RuntimeError("fit must be called first")
        if fold not in (0, 1):
            raise ValueError("fold must be 0 or 1")
        p = self.candidate_[fold] if self.decision_.released else self.base_[fold]
        if indices is None:
            return p.copy()
        idx = np.asarray(indices)
        if (
            idx.ndim != 1
            or not np.issubdtype(idx.dtype, np.integer)
            or np.any(idx < 0)
            or np.any(idx >= len(p))
        ):
            raise ValueError("invalid target indices")
        return p[idx].copy()
