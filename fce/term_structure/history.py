"""Treasury par-rate curve for bootstrapping — real (FRED) or offline synthetic.

Mirrors the driver's offline-first stance: use FRED Treasury constant-maturity
yields when a key is present, otherwise a static but realistic par curve so the
term-structure pillar runs with no network.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# (tenor_years, FRED series id) for the constant-maturity Treasury curve.
_FRED_TENORS = [
    (0.25, "DGS3MO"), (0.5, "DGS6MO"), (1, "DGS1"), (2, "DGS2"),
    (3, "DGS3"), (5, "DGS5"), (7, "DGS7"), (10, "DGS10"), (30, "DGS30"),
]

# A realistic, mildly-inverted static curve (decimal rates) for offline use.
_SYNTHETIC = [
    (0.25, 0.0525), (0.5, 0.0500), (1, 0.0470), (2, 0.0430), (3, 0.0410),
    (5, 0.0400), (7, 0.0410), (10, 0.0420), (30, 0.0440),
]


def synthetic_treasury_curve():
    """Return ``(tenors_years, par_rates)`` for the offline curve."""
    tenors = [t for t, _ in _SYNTHETIC]
    rates = [r for _, r in _SYNTHETIC]
    return tenors, rates


def load_treasury_curve(settings):
    """Return ``(tenors_years, par_rates)`` — FRED latest if keyed, else synthetic."""
    if getattr(settings, "fred_api_key", None):
        try:
            from fredapi import Fred

            fred = Fred(api_key=settings.fred_api_key)
            tenors, rates = [], []
            for ty, sid in _FRED_TENORS:
                s = fred.get_series(sid).dropna()
                if len(s):
                    tenors.append(ty)
                    rates.append(float(s.iloc[-1]) / 100.0)  # FRED yields are in %
            if len(tenors) >= 5:
                return tenors, rates
            logger.warning("FRED curve too sparse (%d pts); using synthetic.", len(tenors))
        except Exception as exc:  # noqa: BLE001 - degrade to offline synthetic
            logger.warning("FRED curve load failed (%s); using synthetic.", exc)
    else:
        logger.info("No FRED_API_KEY set — using synthetic Treasury curve.")
    return synthetic_treasury_curve()
