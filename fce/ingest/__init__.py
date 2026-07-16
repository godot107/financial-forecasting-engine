"""Pillar 0 — free-tier data ingestion (EIA + FRED), cached to pinned parquet.

Adapted from ``energy-batch-trader/energy_trader/eia.py``. Kept project-local
for now; promote to ``../shared/`` once a second consumer needs it (see the
card's Pre-Build Readiness comment).
"""

from fce.ingest.cache import cached_parquet
from fce.ingest.eia import fetch_wti_spot
from fce.ingest.fred import fetch_fred_series

__all__ = ["cached_parquet", "fetch_wti_spot", "fetch_fred_series"]
