"""Pillar 2 — quantitative term-structure & debt engine (QuantLib).

Ingests simulated rate trajectories to bootstrap exact zero-coupon / forward
curves (OIS discounting), then prices corporate debt servicing, floating-rate
amortization (exact day-count: Act/360, Act/Act), and commodity/FX hedge
settlements. The output — per-path discount factors + debt cash flows — is what
Pillar 3 consumes for levered NPV.

Grounding: risk-neutral MC discounting `V0 = e^(−rT)·mean(payoff)` (Hilpisch,
*Python for Finance*, Ch. 16–17) grounds the QuantLib→accounting hand-off before
the heavier bootstrapping.

STUB — see :func:`bootstrap_curve`, :func:`discount_factors`.
"""

from fce.term_structure.curves import bootstrap_curve, discount_factors

__all__ = ["bootstrap_curve", "discount_factors"]
