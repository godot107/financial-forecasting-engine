"""Centralized settings, sourced from environment / ``.env``.

Mirrors the orchestrator-agnostic pattern from ``energy-batch-trader``: read
config in one place so the CLI, a notebook, and any future scheduler all get
identical behavior from the same environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency
    pass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


# Repo root = parent of the ``fce`` package directory.
_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    """Runtime configuration for one pipeline invocation."""

    # --- Data / cache -----------------------------------------------------
    data_dir: Path = field(default_factory=lambda: _ROOT / "data")
    # Pinned data vintage: cached parquet snapshots are keyed by this tag so the
    # 2020/2022 stress backtests reproduce offline regardless of "today".
    data_vintage: str = field(default_factory=lambda: _env("FCE_DATA_VINTAGE", "2026-07"))

    # --- API keys (free tiers) -------------------------------------------
    eia_api_key: str | None = field(default_factory=lambda: _env("EIA_API_KEY"))
    fred_api_key: str | None = field(default_factory=lambda: _env("FRED_API_KEY"))

    # --- Simulation -------------------------------------------------------
    n_paths: int = 10_000  # Monte Carlo driver trajectories
    horizon_months: int = 36  # forward simulation horizon
    seed: int = 42
    hmm_states: int = 2  # regimes in the Pillar-1 driver HMM (calm / crisis)

    # --- Discounting / accounting (MVP defaults) -------------------------
    tax_rate: float = 0.23  # effective corporate tax (EDGAR comparables ~21-25%)
    wacc: float = 0.10  # fallback flat discount when QuantLib curve absent

    # --- Optimization -----------------------------------------------------
    capex_budget: float = 200e6  # annual CapEx ceiling ($)
    cfar_floor: float = 200e6  # CFaR_95 guaranteed tail-liquidity floor ($)
    cfar_alpha: float = 0.95  # Cash-Flow-at-Risk confidence level

    # --- Backtest ---------------------------------------------------------
    embargo_frac: float = 0.01  # López de Prado embargo (fraction of bars)

    def vintage_path(self, name: str) -> Path:
        """Cache path for a named parquet snapshot at the pinned vintage."""
        return self.data_dir / f"{name}.{self.data_vintage}.parquet"


def get_settings() -> Settings:
    """Build a fresh :class:`Settings` from the current environment."""
    return Settings()
