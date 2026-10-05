"""SafeDelta: split-control diagnostics and calibration-only release decisions for few-shot perturbation-response adaptation."""

from .adapter import ReleaseDecision, SplitControlAdapter
from .core import (
    centre,
    check_identity,
    dual_regime_scores,
    inflation_from_theory,
    inner,
    measure_a,
    measure_a_directional,
    measure_lambda,
    pcc,
    predicted_gap,
    split_control,
)

__version__ = "1.0.0"
__all__ = [
    "centre",
    "pcc",
    "inner",
    "split_control",
    "measure_a",
    "measure_a_directional",
    "measure_lambda",
    "predicted_gap",
    "log_normalize_counts",
    "pseudobulk_mean",
    "dual_regime_scores",
    "check_identity",
    "inflation_from_theory",
    "SplitControlAdapter",
    "ReleaseDecision",
]

from .preprocessing import log_normalize_counts, pseudobulk_mean
