"""Floating-rate corporate debt servicing (Pillar 2).

Turns a simulated short-rate path into per-scenario monthly interest expense on a
project's debt: ``interest = outstanding · (short_rate + credit_spread) · τ``,
where ``τ`` is the exact **Actual/360** accrual fraction for each month (QuantLib
day count) — so debt service is genuinely rate-sensitive across scenarios, and a
rate spike widens interest exactly when crude-exposed revenue is weakest.

Amortization: ``bullet`` (constant notional) or ``straight`` (linear to zero).
"""

from __future__ import annotations

import numpy as np
import QuantLib as ql


def monthly_accruals(n_months: int, valuation_date=None) -> np.ndarray:
    """Actual/360 accrual fractions for the next ``n_months`` months, ``(n,)``."""
    dc = ql.Actual360()
    start = (
        ql.Date(valuation_date.day, valuation_date.month, valuation_date.year)
        if valuation_date is not None and not isinstance(valuation_date, ql.Date)
        else (valuation_date or ql.Date(30, 6, 2026))
    )
    fracs = []
    d0 = start
    for m in range(1, n_months + 1):
        d1 = start + ql.Period(m, ql.Months)
        fracs.append(dc.yearFraction(d0, d1))
        d0 = d1
    return np.array(fracs)


def outstanding_schedule(notional: float, n_months: int, amortize: str = "bullet") -> np.ndarray:
    """Outstanding principal at the start of each month, ``(n_months,)``."""
    if amortize == "bullet":
        return np.full(n_months, float(notional))
    if amortize == "straight":
        # Linear paydown: full at month 1 → ~0 after the last month.
        return float(notional) * (1.0 - np.arange(n_months) / n_months)
    raise ValueError(f"unknown amortize scheme: {amortize!r}")


def floating_rate_interest(
    short_rates: np.ndarray,
    *,
    notional: float,
    credit_spread: float,
    amortize: str = "bullet",
    valuation_date=None,
) -> np.ndarray:
    """Per-scenario monthly interest expense, ``(S, T)`` (matches ``short_rates``).

    ``interest_{s,t} = outstanding_t · (r_{s,t} + credit_spread) · τ_t`` with
    Actual/360 ``τ_t``.
    """
    r = np.asarray(short_rates, float)
    s, t = r.shape
    accr = monthly_accruals(t, valuation_date)             # (T,)
    outstanding = outstanding_schedule(notional, t, amortize)  # (T,)
    coupon = r + credit_spread
    return coupon * outstanding[None, :] * accr[None, :]
