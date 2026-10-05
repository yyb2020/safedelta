"""Count-to-expression conversion, with no implicit filtering or gene selection."""

import numpy as np

from .core import _array


def log_normalize_counts(counts, target_sum=10000.0):
    """Dense cells × genes: log1p(count / full-library sum * target_sum).

    Call BEFORE selecting genes. Zero-library cells are rejected, not silently
    dropped. Sparse datasets should be processed in bounded dense blocks.
    Already normalized expression must not pass through this function again.
    """
    x = _array(counts, ndim=2, name="counts")
    if np.any(x < 0) or not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("nonnegative counts and a positive target_sum required")
    total = x.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("zero-library cells must be filtered explicitly")
    return np.log1p(x / total[:, None] * target_sum)


def pseudobulk_mean(expression, groups):
    """Return (sorted group labels, arithmetic means of cell expression).

    This is mean(log-normalized cells), not log(normalized summed counts).
    """
    x = _array(expression, ndim=2, name="expression")
    labels = np.asarray(groups)
    if labels.ndim != 1 or len(labels) != len(x):
        raise ValueError("one group label per cell required")
    if any(
        v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))
        for v in labels
    ):
        raise ValueError("missing group labels")
    names, inverse = np.unique(labels.astype(str), return_inverse=True)
    result = np.zeros((len(names), x.shape[1]), dtype=float)
    np.add.at(result, inverse, x)
    result /= np.bincount(inverse)[:, None]
    return names, result
