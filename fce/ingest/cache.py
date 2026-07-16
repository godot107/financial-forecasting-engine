"""Pinned-vintage parquet cache.

A thin decorator: the first call fetches live and writes ``<name>.<vintage>.parquet``;
every later call for the same vintage reads the snapshot. This is what makes the
2020/2022 stress backtests reproducible offline — the data vintage is frozen, so
re-running months later yields byte-identical inputs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def cached_parquet(path: Path, fetch, *, refresh: bool = False) -> pd.DataFrame:
    """Return ``fetch()`` output, reading/writing a parquet snapshot at ``path``.

    ``fetch`` is a zero-arg callable returning a DataFrame or Series. Pass
    ``refresh=True`` to force a re-fetch and overwrite the snapshot.
    """
    path = Path(path)
    if path.exists() and not refresh:
        logger.info("cache hit: %s", path.name)
        return pd.read_parquet(path)

    logger.info("cache miss: fetching %s", path.name)
    obj = fetch()
    frame = obj.to_frame() if isinstance(obj, pd.Series) else obj
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame
