"""Tests for the Pillar-2 QuantLib term structure, short rates, and debt."""

from __future__ import annotations

import numpy as np

from fce.config import Settings
from fce.projects import simulate_project_scenarios
from fce.term_structure import (
    bootstrap_curve,
    discount_factors_from_short_rates,
    floating_rate_interest,
    monthly_accruals,
    rate_scenario_paths,
    simulate_short_rates,
    synthetic_treasury_curve,
)


def test_bootstrapped_curve_discounts_are_monotone():
    tenors, par = synthetic_treasury_curve()
    curve = bootstrap_curve(tenors, par)
    dfs = curve.monthly_discount_factors(120)
    assert np.all(dfs > 0) and np.all(dfs <= 1.0 + 1e-9)
    assert np.all(np.diff(dfs) < 0)                     # strictly decreasing
    # Zero rate at the short end is near the input front par rate (~5.25%).
    assert 0.03 < curve.front_rate() < 0.07


def test_curve_reprices_inputs_roughly():
    tenors, par = synthetic_treasury_curve()
    curve = bootstrap_curve(tenors, par)
    # The 5y zero rate should sit in the neighborhood of the 5y par input (4.0%).
    z5 = curve.zero_rate(60)
    assert abs(z5 - 0.040) < 0.01


def test_vasicek_short_rates_mean_revert():
    sr = simulate_short_rates(n_paths=4000, horizon=60, r0=0.052, theta=0.040,
                              kappa=0.5, sigma=0.008, seed=0)
    assert sr.shape == (4000, 60)
    assert np.all(sr >= 0.0)                             # floored
    # Starts near r0, drifts toward theta.
    assert abs(sr[:, 0].mean() - 0.052) < 0.01
    assert sr[:, -1].mean() < sr[:, 0].mean()            # reverting down toward theta


def test_discount_factors_from_short_rates():
    sr = simulate_short_rates(n_paths=1000, horizon=36, r0=0.05, theta=0.05, seed=1)
    df = discount_factors_from_short_rates(sr, spread=0.05)
    assert df.shape == (1000, 36)
    assert np.all((df > 0) & (df <= 1.0))
    assert np.all(np.diff(df, axis=1) < 0)              # decreasing along horizon


def test_monthly_accruals_are_act360():
    accr = monthly_accruals(12)
    assert accr.shape == (12,)
    # Each month ≈ 30/360; a year of them sums to ≈ 365/360.
    assert np.all((accr > 0.07) & (accr < 0.10))
    assert abs(accr.sum() - 365 / 360) < 0.02


def test_floating_interest_rises_with_rates_and_amortizes():
    sr = simulate_short_rates(n_paths=3000, horizon=36, r0=0.05, theta=0.05,
                              sigma=0.012, seed=2)
    bullet = floating_rate_interest(sr, notional=60e6, credit_spread=0.02, amortize="bullet")
    assert bullet.shape == sr.shape
    hi = sr[:, -1] > np.quantile(sr[:, -1], 0.9)
    lo = sr[:, -1] < np.quantile(sr[:, -1], 0.1)
    assert bullet[hi].mean() > bullet[lo].mean()        # rate-sensitive
    # Straight amortization pays less total interest than bullet (principal falls).
    straight = floating_rate_interest(sr, notional=60e6, credit_spread=0.02, amortize="straight")
    assert straight.sum() < bullet.sum()


def test_rate_paths_feed_allocation_scenarios():
    s = Settings(n_paths=800, horizon_months=24)
    short_rates, curve = rate_scenario_paths(s)
    assert short_rates.shape == (800, 24)
    scen = simulate_project_scenarios(s, short_rates=short_rates)
    assert scen.npv_per_dollar.shape == (800, 5)
    assert np.isfinite(scen.npv_per_dollar).all()
    # Term-structure discounting changes NPV vs. the flat-WACC baseline.
    flat = simulate_project_scenarios(s)
    assert not np.allclose(scen.npv_per_dollar, flat.npv_per_dollar)
