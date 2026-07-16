"""QuantLib yield-curve bootstrapping and discounting (Pillar 2).

STUB. Target interface below.

Until QuantLib is wired in, Pillar 3 falls back to a flat ``Settings.wacc``
discount (see :func:`fce.accounting.statements.npv`), which is enough to validate
the accounting articulation end-to-end on the MVP slice.
"""

from __future__ import annotations

import numpy as np


def bootstrap_curve(quotes, *, valuation_date, day_count: str = "Act/360"):
    """Bootstrap a zero-coupon discount curve from market ``quotes``.

    Returns a curve object exposing zero rates, forward rates, and discount
    factors. Intended to wrap ``QuantLib.PiecewiseLogLinearDiscount`` or similar.
    """
    raise NotImplementedError("Pillar 2 (QuantLib bootstrap) not implemented yet.")


def discount_factors(curve, times: np.ndarray) -> np.ndarray:
    """Discount factors from ``curve`` at year-fractions ``times``."""
    raise NotImplementedError("Pillar 2 (QuantLib discounting) not implemented yet.")
