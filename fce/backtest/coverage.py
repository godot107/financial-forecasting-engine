"""Quantile calibration metrics: interval coverage and pinball loss.

These score the *probabilistic* driver forecasts (statistical loop), not the
allocation policy. We optimize pinball/quantile loss rather than MSE because the
whole product sells prediction *intervals*, not point estimates. Empirical
coverage verifies the intervals are honest: realized values should fall inside
``[P10, P90]`` ~80 % of the time.
"""

from __future__ import annotations

import numpy as np


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of ``actual`` values falling within ``[lower, upper]`` (0..1)."""
    actual = np.asarray(actual, float)
    inside = (actual >= np.asarray(lower, float)) & (actual <= np.asarray(upper, float))
    return float(np.mean(inside))


def pinball_loss(actual: np.ndarray, pred_q: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss of a ``q``-quantile forecast.

    ``q`` in (0, 1). Under-prediction is penalized by ``q``, over-prediction by
    ``1 - q`` — the asymmetric loss whose minimizer is the true ``q``-quantile.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    actual = np.asarray(actual, float)
    pred_q = np.asarray(pred_q, float)
    err = actual - pred_q
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))
