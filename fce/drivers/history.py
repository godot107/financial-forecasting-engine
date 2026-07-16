"""WTI monthly-price history for fitting the driver model.

Uses the real EIA series when a key is available (cached to pinned parquet),
and otherwise falls back to a **synthetic two-regime series** so the whole
pipeline — and the tutorial notebook — runs offline and reproducibly. The
synthetic generator is a genuine Gaussian HMM, which doubles as ground truth for
the model-recovery test (``tests/test_drivers.py``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A calm regime and a crisis regime (monthly log-return mean / vol).
_TRUE_LOCS = np.array([0.004, -0.010])
_TRUE_SCALES = np.array([0.035, 0.130])
_TRUE_TRANS = np.array([[0.94, 0.06],   # calm tends to persist
                        [0.20, 0.80]])  # crises are stickier-than-random but exit
_TRUE_INIT = np.array([0.85, 0.15])


def simulate_true_hmm(n: int, *, seed: int = 0):
    """Generate ``n`` monthly log-returns from the known two-regime HMM.

    Returns ``(returns, states)``. Ground truth for recovery tests and the
    synthetic history.
    """
    rng = np.random.default_rng(seed)
    states = np.empty(n, dtype=int)
    returns = np.empty(n)
    s = rng.choice(2, p=_TRUE_INIT)
    for t in range(n):
        states[t] = s
        returns[t] = rng.normal(_TRUE_LOCS[s], _TRUE_SCALES[s])
        s = rng.choice(2, p=_TRUE_TRANS[s])
    return returns, states


def synthetic_wti_monthly(n_months: int = 420, *, seed: int = 0, s0: float = 30.0) -> pd.Series:
    """A synthetic monthly WTI price series (compounded from the true HMM)."""
    returns, _ = simulate_true_hmm(n_months, seed=seed)
    prices = s0 * np.exp(np.cumsum(returns))
    idx = pd.date_range(end="2026-06-01", periods=n_months, freq="MS")
    return pd.Series(prices, index=idx, name="wti")


def load_wti_monthly(settings, *, n_months: int = 420) -> pd.Series:
    """Monthly WTI prices: real (EIA, cached) if a key is set, else synthetic.

    Real path resamples EIA daily spot to month-start last-observation. Degrades
    to the synthetic series on any failure so nothing downstream needs a network.
    """
    if getattr(settings, "eia_api_key", None):
        try:
            from fce.ingest import cached_parquet
            from fce.ingest.eia import fetch_wti_spot

            path = settings.vintage_path("wti_daily")
            daily = cached_parquet(path, lambda: fetch_wti_spot(settings.eia_api_key))
            series = daily.iloc[:, 0] if hasattr(daily, "iloc") else daily
            monthly = series.resample("MS").last().dropna()
            if len(monthly) >= 60:
                return monthly.tail(n_months).rename("wti")
            logger.warning("EIA WTI history too short (%d months); using synthetic.", len(monthly))
        except Exception as exc:  # noqa: BLE001 - degrade to offline synthetic
            logger.warning("EIA WTI load failed (%s); using synthetic history.", exc)
    else:
        logger.info("No EIA_API_KEY set — using synthetic WTI history.")
    return synthetic_wti_monthly(n_months, seed=settings.seed)
