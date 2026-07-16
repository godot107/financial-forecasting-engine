"""EIA Open Data API v2 — WTI (Cushing) daily spot price, the MVP macro driver.

Lifted from ``energy-batch-trader/energy_trader/eia.py`` (``fetch_wti_spot``) and
trimmed to just the series this project needs. WTI feeds the Pillar-1 driver
model (regime detection + Monte Carlo trajectories) and, through the accounting
engine, the synthetic energy projects' revenue.

Docs: https://www.eia.gov/opendata/documentation.php
(The API key must be in the URL; EIA ignores it in HTTP headers.)
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

EIA_V2_BASE = "https://api.eia.gov/v2"
WTI_SPOT_ROUTE = "petroleum/pri/spt/data"
WTI_SPOT_SERIES = "RWTC"  # Cushing, OK WTI spot price FOB, $/bbl (daily)


def fetch_wti_spot(api_key: str, length: int = 5000) -> pd.Series:
    """Return daily WTI (Cushing) spot price, chronological (oldest→newest).

    Raises on transport/HTTP errors so the caller (or cache layer) can decide how
    to degrade. ``length`` caps the number of most-recent observations pulled.
    """
    if not api_key:
        raise ValueError("EIA_API_KEY is required to fetch WTI spot prices")

    params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "facets[series][]": WTI_SPOT_SERIES,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": length,
    }
    resp = requests.get(f"{EIA_V2_BASE}/{WTI_SPOT_ROUTE}", params=params, timeout=30)
    resp.raise_for_status()
    rows = [r for r in resp.json()["response"]["data"] if r.get("value") is not None]
    rows = list(reversed(rows))  # oldest→newest
    return pd.Series(
        [float(r["value"]) for r in rows],
        index=pd.to_datetime([r["period"] for r in rows]),
        name="wti_spot",
    )
