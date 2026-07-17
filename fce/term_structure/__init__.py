"""Pillar 2 — quantitative term-structure & debt engine (QuantLib).

Ingests simulated rate trajectories to bootstrap exact zero-coupon / forward
curves (OIS discounting), then prices corporate debt servicing, floating-rate
amortization (exact day-count: Act/360, Act/Act), and commodity/FX hedge
settlements. The output — per-path discount factors + debt cash flows — is what
Pillar 3 consumes for levered NPV.

Grounding: risk-neutral MC discounting `V0 = e^(−rT)·mean(payoff)` (Hilpisch,
*Python for Finance*, Ch. 16–17) grounds the QuantLib→accounting hand-off before
the heavier bootstrapping.

Built: :func:`bootstrap_curve` (QuantLib piecewise bootstrap → :class:`TermStructure`),
Vasicek short-rate simulation (:func:`simulate_short_rates`), pathwise discounting
(:func:`discount_factors_from_short_rates`), and floating-rate debt service
(:func:`floating_rate_interest`). :func:`rate_scenario_paths` ties them together
for the allocation pipeline.
"""

from fce.term_structure.curves import (
    TermStructure,
    bootstrap_curve,
    discount_factors,
)
from fce.term_structure.debt import floating_rate_interest, monthly_accruals
from fce.term_structure.history import (
    load_treasury_curve,
    synthetic_treasury_curve,
)
from fce.term_structure.rates import (
    discount_factors_from_short_rates,
    rate_scenario_paths,
    simulate_short_rates,
)

__all__ = [
    "TermStructure", "bootstrap_curve", "discount_factors",
    "simulate_short_rates", "discount_factors_from_short_rates",
    "rate_scenario_paths", "floating_rate_interest", "monthly_accruals",
    "load_treasury_curve", "synthetic_treasury_curve",
]
