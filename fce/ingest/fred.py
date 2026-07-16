"""FRED (Federal Reserve Economic Data) — rates, credit spreads, inflation.

Free via the ``fredapi`` client. These series feed two pillars:

* **Treasury yields** (``DGS1MO``…``DGS30``) → QuantLib zero-coupon bootstrap and
  floating-rate debt servicing (Pillar 2).
* **Inflation** (``PPIACO``, ``CPIAUCSL``) → OpEx/CapEx cost escalation inside the
  deterministic accounting engine (Pillar 3).
* **Credit spreads** (``BAMLC0A0CM``, ``BAMLH0A0HYM2``) → discount-rate add-ons.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical series used across the engine. Extend as pillars come online.
DEFAULT_SERIES: dict[str, str] = {
    "DGS3MO": "ust_3m",
    "DGS2": "ust_2y",
    "DGS10": "ust_10y",
    "BAMLC0A0CM": "ig_oas",  # IG corporate option-adjusted spread
    "PPIACO": "ppi_all",
    "CPIAUCSL": "cpi",
}


def fetch_fred_series(
    api_key: str, series: dict[str, str] | None = None
) -> pd.DataFrame:
    """Return a wide DataFrame of FRED series (columns renamed to friendly names).

    ``series`` maps FRED series id → column name; defaults to
    :data:`DEFAULT_SERIES`. Requires the ``fredapi`` package.
    """
    if not api_key:
        raise ValueError("FRED_API_KEY is required to fetch FRED series")

    from fredapi import Fred  # lazy import: keeps the base install lean

    series = series or DEFAULT_SERIES
    fred = Fred(api_key=api_key)
    cols = {name: fred.get_series(sid) for sid, name in series.items()}
    frame = pd.DataFrame(cols)
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    return frame.sort_index()
