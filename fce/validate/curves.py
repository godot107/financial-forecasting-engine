"""Term-structure validation: does the bootstrapped curve reprice its inputs?

A yield curve that cannot reprice the very instruments it was calibrated to is
broken — this is the most basic internal-consistency check in fixed income, and
it needs no market data (it is a pure round-trip). We rebuild each calibration
instrument, price it *off the bootstrapped curve*, and measure the residual
against the input par quote in basis points.

- **Deposits** (tenor ≤ 1y): the curve's implied simple Act/360 rate between spot
  and the deposit maturity should equal the input deposit rate.
- **Swaps** (tenor > 1y): the par (fair) fixed rate of a vanilla swap priced on
  the curve should equal the input swap rate.

Grounding (textbook-kb): bootstrap calibration and repricing — James Ma Weiming,
*Mastering Python for Finance*, Ch. 5 (pp. 158–167); a bootstrap "proceeds out
through the valuation horizon" fitting each instrument in turn — Swindle,
*Valuation and Risk Management in Energy Markets*, §7 (p. 165).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import QuantLib as ql

from fce.term_structure.curves import TermStructure, bootstrap_curve


@dataclass
class RepricingResult:
    """Repricing residuals for each calibration instrument (in basis points)."""

    tenors_years: np.ndarray
    par_rates: np.ndarray
    implied_rates: np.ndarray
    kinds: list[str]            # "deposit" | "swap" per instrument

    @property
    def residuals_bp(self) -> np.ndarray:
        """Signed repricing error per instrument, in basis points."""
        return (self.implied_rates - self.par_rates) * 1e4

    @property
    def max_abs_bp(self) -> float:
        return float(np.max(np.abs(self.residuals_bp)))

    def to_markdown(self) -> str:
        lines = [
            "### Curve repricing round-trip",
            "",
            "| Instrument | Tenor | Par quote | Repriced | Residual (bp) |",
            "|---|---:|---:|---:|---:|",
        ]
        for ty, kind, par, imp, bp in zip(
            self.tenors_years, self.kinds, self.par_rates,
            self.implied_rates, self.residuals_bp,
        ):
            label = f"{kind} {ty:g}y"
            lines.append(
                f"| {label} | {ty:g}y | {par*100:.3f}% | {imp*100:.3f}% | {bp:+.3f} |"
            )
        lines.append("")
        lines.append(f"**Max abs residual: {self.max_abs_bp:.3f} bp** "
                     "(a healthy bootstrap reprices to well under 1 bp).")
        return "\n".join(lines)


def reprice_curve(tenors_years, par_rates, *, valuation_date=None) -> RepricingResult:
    """Bootstrap a curve from ``(tenor, par_rate)`` and reprice each input.

    Returns a :class:`RepricingResult`; ``residuals_bp`` should be ~0 (well under
    1 bp) for a correctly bootstrapped curve.
    """
    tenors_years = [float(t) for t in tenors_years]
    par_rates = [float(r) for r in par_rates]
    ts = bootstrap_curve(tenors_years, par_rates, valuation_date=valuation_date)
    return _reprice_against(ts, tenors_years, par_rates)


def _reprice_against(ts: TermStructure, tenors_years, par_rates) -> RepricingResult:
    # Reproduce the conventions used in bootstrap_curve so the round-trip is exact.
    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    dc = ql.Actual360()
    ql.Settings.instance().evaluationDate = ts.valuation
    handle = ql.YieldTermStructureHandle(ts.curve)
    index = ql.IborIndex(
        "GenIdx", ql.Period(3, ql.Months), 2, ql.USDCurrency(),
        cal, ql.ModifiedFollowing, False, dc, handle,
    )
    settle = cal.advance(ts.valuation, 2, ql.Days)

    implied, kinds = [], []
    for ty, _rate in zip(tenors_years, par_rates):
        if ty <= 1.0:
            maturity = cal.advance(
                settle, ql.Period(int(round(ty * 12)), ql.Months), ql.ModifiedFollowing
            )
            tau = dc.yearFraction(settle, maturity)
            df_s = ts.curve.discount(settle)
            df_m = ts.curve.discount(maturity)
            implied.append((df_s / df_m - 1.0) / tau)   # simple Act/360 deposit rate
            kinds.append("deposit")
        else:
            swap = ql.MakeVanillaSwap(
                ql.Period(int(round(ty)), ql.Years), index, 0.0, ql.Period(0, ql.Days),
                fixedLegTenor=ql.Period(1, ql.Years),
                fixedLegDayCount=dc,
                pricingEngine=ql.DiscountingSwapEngine(handle),
            )
            implied.append(float(swap.fairRate()))
            kinds.append("swap")

    return RepricingResult(
        tenors_years=np.asarray(tenors_years),
        par_rates=np.asarray(par_rates),
        implied_rates=np.asarray(implied),
        kinds=kinds,
    )
