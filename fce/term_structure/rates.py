"""Stochastic short-rate simulation (Vasicek) + path discounting (Pillar 2).

Simulates monthly short-rate trajectories with a mean-reverting one-factor Vasicek
model, calibrated so the long-run mean sits at the curve's long end and today's
rate at the curve's front. These paths drive both floating-rate debt service
(:mod:`fce.term_structure.debt`) and per-scenario discounting of FCFF — so a rate
shock feeds the whole engine, exactly as the hybrid design intends.

Grounding: Vasicek short-rate model & zero-coupon pricing — James Ma Weiming,
*Mastering Python for Finance*, Ch. 5 (pp. 164–167); risk-neutral Monte-Carlo
discounting ``V0 = E[e^{-∫r dt} · CF]`` — Hilpisch, *Python for Finance*, Ch. 16–17.
"""

from __future__ import annotations

import numpy as np


def simulate_short_rates(
    *,
    n_paths: int,
    horizon: int,
    r0: float,
    theta: float,
    kappa: float = 0.30,
    sigma: float = 0.010,
    seed: int = 0,
    dt: float = 1.0 / 12.0,
    floor: float = 0.0,
) -> np.ndarray:
    """Vasicek short-rate paths, ``(n_paths, horizon)`` (monthly).

    ``dr = kappa(theta − r)dt + sigma dW``. ``r0`` is today's short rate, ``theta``
    the long-run mean (both naturally taken from the bootstrapped curve). Rates are
    floored at ``floor`` to avoid pathological negatives in the accounting layer.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_paths, horizon))
    r = np.full(n_paths, float(r0))
    sqrt_dt = np.sqrt(dt)
    for t in range(horizon):
        dw = rng.normal(0.0, sqrt_dt, size=n_paths)
        r = r + kappa * (theta - r) * dt + sigma * dw
        r = np.maximum(r, floor)
        out[:, t] = r
    return out


def discount_factors_from_short_rates(
    short_rates: np.ndarray, *, spread: float = 0.0, dt: float = 1.0 / 12.0
) -> np.ndarray:
    """Per-path discount factors from a short-rate path, ``(S, T)``.

    ``DF_{s,t} = exp(-Σ_{k≤t} (r_{s,k} + spread) · dt)`` — the pathwise
    money-market discount. ``spread`` lifts the risk-free short rate toward a
    project discount rate (a coarse WACC proxy until a full CAPM build).
    """
    rates = np.asarray(short_rates, float) + spread
    return np.exp(-np.cumsum(rates * dt, axis=1))


def rate_scenario_paths(settings, *, kappa: float = 0.30, sigma: float = 0.010):
    """Bootstrap the curve and simulate short-rate paths → ``(short_rates, curve)``.

    Calibrates today's rate ``r0`` to the curve's short end and the long-run mean
    ``theta`` to its 10y zero — so the Vasicek paths start at, and revert toward,
    the observed term structure. Analogue of ``drivers.wti_scenario_paths``.
    """
    from fce.term_structure.curves import bootstrap_curve
    from fce.term_structure.history import load_treasury_curve

    tenors, par = load_treasury_curve(settings)
    curve = bootstrap_curve(tenors, par)
    r0 = curve.front_rate()
    theta = curve.zero_rate(120)  # 10y zero as the long-run mean
    short_rates = simulate_short_rates(
        n_paths=settings.n_paths, horizon=settings.horizon_months,
        r0=r0, theta=theta, kappa=kappa, sigma=sigma, seed=settings.seed,
    )
    return short_rates, curve
