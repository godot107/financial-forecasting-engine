"""QuantLib yield-curve bootstrapping and discounting (Pillar 2).

Bootstraps a discount term structure from market par rates — deposit rate helpers
at the short end, swap rate helpers further out — into a
``PiecewiseLogLinearDiscount`` curve, then exposes discount factors, zero rates,
and forward rates with exact day-count conventions.

Grounding (textbook-kb): yield-curve bootstrapping / spot & forward rates —
James Ma Weiming, *Mastering Python for Finance*, Ch. 5 (pp. 138–167);
forward↔zero-coupon relationship — Berk & DeMarzo, *Corporate Finance* (2024),
Appendix 6A (p. 244). QuantLib does the numerics here rather than a hand-rolled
bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import QuantLib as ql


def _to_ql_date(d) -> "ql.Date":
    if isinstance(d, ql.Date):
        return d
    return ql.Date(d.day, d.month, d.year)


@dataclass
class TermStructure:
    """A bootstrapped discount curve + convenience extractors (monthly grid)."""

    curve: ql.YieldTermStructure
    valuation: ql.Date
    day_count: ql.DayCounter

    def _date(self, months: int) -> ql.Date:
        return self.valuation + ql.Period(int(months), ql.Months)

    def discount(self, months: int) -> float:
        """Discount factor at ``months`` from valuation."""
        return float(self.curve.discount(self._date(months)))

    def monthly_discount_factors(self, n_months: int) -> np.ndarray:
        """Discount factors for months ``1..n_months`` → shape ``(n_months,)``."""
        return np.array([self.discount(m) for m in range(1, n_months + 1)])

    def zero_rate(self, months: int, comp=ql.Continuous) -> float:
        """Continuously-compounded zero rate at ``months``."""
        return float(self.curve.zeroRate(self._date(months), self.day_count, comp).rate())

    def forward_rate(self, months: int, tenor_months: int = 12, comp=ql.Continuous) -> float:
        """Forward rate from ``months`` over ``tenor_months``."""
        d1, d2 = self._date(months), self._date(months + tenor_months)
        return float(self.curve.forwardRate(d1, d2, self.day_count, comp).rate())

    def front_rate(self) -> float:
        """Short-end zero rate (~3M) — a natural ``r0`` for short-rate simulation."""
        return self.zero_rate(3)


def bootstrap_curve(
    tenors_years,
    par_rates,
    *,
    valuation_date=None,
) -> TermStructure:
    """Bootstrap a discount curve from ``(tenor_years, par_rate)`` quotes.

    Tenors ≤ 1y become deposit helpers, longer tenors become swap helpers, fed to
    a ``PiecewiseLogLinearDiscount`` bootstrap. Returns a :class:`TermStructure`.
    """
    tenors_years = list(tenors_years)
    par_rates = list(par_rates)
    val = _to_ql_date(valuation_date) if valuation_date is not None else ql.Date(30, 6, 2026)
    ql.Settings.instance().evaluationDate = val

    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    dc = ql.Actual360()
    index = ql.IborIndex(
        "GenIdx", ql.Period(3, ql.Months), 2, ql.USDCurrency(),
        cal, ql.ModifiedFollowing, False, dc,
    )

    helpers = []
    for ty, rate in zip(tenors_years, par_rates):
        q = ql.QuoteHandle(ql.SimpleQuote(float(rate)))
        if ty <= 1.0:
            helpers.append(ql.DepositRateHelper(
                q, ql.Period(int(round(ty * 12)), ql.Months), 2,
                cal, ql.ModifiedFollowing, False, dc,
            ))
        else:
            helpers.append(ql.SwapRateHelper(
                q, ql.Period(int(round(ty)), ql.Years), cal,
                ql.Annual, ql.ModifiedFollowing, dc, index,
            ))

    curve = ql.PiecewiseLogLinearDiscount(val, helpers, dc)
    curve.enableExtrapolation()
    return TermStructure(curve=curve, valuation=val, day_count=dc)


def discount_factors(curve: TermStructure, months) -> np.ndarray:
    """Discount factors at an array of month offsets."""
    return np.array([curve.discount(int(m)) for m in np.atleast_1d(months)])
