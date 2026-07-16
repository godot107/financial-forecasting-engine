"""THE orchestrator-agnostic entrypoint (``run_pipeline``).

Mirrors the ``energy-batch-trader`` convention: all logic lives here so a CLI, a
notebook, or a future scheduler just calls ``run_pipeline()`` — never duplicate
logic into an orchestrator.

Two entrypoints:

* :func:`run_pipeline` — the single-project MVP accounting slice (Pillar 3),
  proving the deterministic core balances end-to-end.
* :func:`run_allocation` — the full capital-allocation decision (Pillars 3+4):
  simulate all five projects → CVXPY allocate under a CFaR floor → trace the
  risk-return frontier. This is what the dashboard/deck consume.

Both still ride a *placeholder* driver for revenue (build order:
accounting-before-drivers). Swap it for :func:`fce.drivers.simulate_drivers`
once Pillar 1 lands; swap the flat WACC for the QuantLib curve once Pillar 2 lands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from fce.accounting import assert_balanced, build_statements, npv
from fce.config import Settings, get_settings
from fce.optimize import Allocation, allocate, efficient_frontier, max_cfar
from fce.projects import PROJECTS, simulate_project_scenarios

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    npv_p10: float
    npv_p50: float
    npv_p90: float
    n_paths: int
    horizon_months: int
    balanced: bool


def _placeholder_wti_paths(settings: Settings) -> np.ndarray:
    """Toy geometric-Brownian-motion WTI trajectories, ``(P, T)``.

    Stands in for Pillar 1 (NumPyro HMM). Grounds the MVP so the accounting layer
    has something to consume; NOT a modeling claim — see fce/drivers.
    """
    rng = np.random.default_rng(settings.seed)
    p, t = settings.n_paths, settings.horizon_months
    mu, sigma, s0 = 0.0, 0.08, 75.0  # monthly drift/vol, $/bbl start
    shocks = rng.normal(mu - 0.5 * sigma**2, sigma, size=(p, t))
    return s0 * np.exp(np.cumsum(shocks, axis=1))


def run_pipeline(settings: Settings | None = None) -> PipelineResult:
    """Run the MVP capital-forecasting slice and return NPV distribution stats."""
    settings = settings or get_settings()
    p, t = settings.n_paths, settings.horizon_months

    # 1. Driver (placeholder). WTI → revenue for one synthetic energy project:
    #    a refinery whose monthly revenue scales with the crude price.
    wti = _placeholder_wti_paths(settings)
    barrels_per_month = 1.0e6
    revenue = wti * barrels_per_month

    # 2. Deterministic accounting articulation (fixed operating assumptions).
    opex = 0.55 * revenue
    depreciation = np.full((p, t), 3.0e6)
    capex = np.full((p, t), 2.5e6)
    delta_nwc = 0.02 * revenue
    interest = np.full((p, t), 1.0e6)
    net_debt_issued = np.zeros((p, t))

    stmt = build_statements(
        revenue=revenue,
        opex=opex,
        depreciation=depreciation,
        capex=capex,
        delta_nwc=delta_nwc,
        interest=interest,
        net_debt_issued=net_debt_issued,
        tax_rate=settings.tax_rate,
        opening_cash=50.0e6,
        opening_ppe=300.0e6,
        opening_debt=120.0e6,
    )

    # 3. Guard the auditability promise every run, not just in tests.
    assert_balanced(stmt, opening_cash=50.0e6)

    # 4. Discount FCFF to NPV (flat WACC until QuantLib term structure lands).
    project_npv = npv(stmt.fcff, settings.wacc)
    p10, p50, p90 = np.quantile(project_npv, [0.1, 0.5, 0.9])

    logger.info(
        "MVP NPV distribution ($M): P10=%.1f  P50=%.1f  P90=%.1f (%d paths)",
        p10 / 1e6, p50 / 1e6, p90 / 1e6, p,
    )
    return PipelineResult(
        npv_p10=float(p10),
        npv_p50=float(p50),
        npv_p90=float(p90),
        n_paths=p,
        horizon_months=t,
        balanced=True,
    )


@dataclass
class AllocationResult:
    """Full capital-allocation decision + the risk-return frontier behind it."""

    allocation: Allocation
    frontier: list  # list[FrontierPoint]
    budget: float
    cfar_floor: float


def run_allocation(
    settings: Settings | None = None,
    *,
    cfar_floor: float | None = None,
    n_frontier: int = 12,
    use_hmm: bool = False,
) -> AllocationResult:
    """Simulate the 5 projects and solve the CFaR-constrained allocation (Pillars 3+4).

    ``cfar_floor`` defaults to ``Settings.cfar_floor``. ``use_hmm=True`` drives the
    projects with the Pillar-1 NumPyro HMM posterior-predictive WTI paths (cached
    posterior) instead of the placeholder GBM. Also traces the efficient frontier
    (expected NPV vs. guaranteed liquidity) — the data the dashboard/deck plot.
    Returns an :class:`AllocationResult`.
    """
    settings = settings or get_settings()
    floor = settings.cfar_floor if cfar_floor is None else cfar_floor

    if use_hmm:
        from fce.drivers import wti_scenario_paths

        wti_paths, wti0 = wti_scenario_paths(settings)
        scen = simulate_project_scenarios(settings, wti_paths=wti_paths, wti0=wti0)
    else:
        scen = simulate_project_scenarios(settings)
    caps = np.array([p.invested for p in PROJECTS])

    alloc = allocate(
        scen.npv_per_dollar, scen.liq_per_dollar,
        names=scen.names, caps=caps, budget=settings.capex_budget,
        cfar_floor=floor, alpha=settings.cfar_alpha,
    )

    # Frontier endpoints: the max-NPV book gives the LOWEST guaranteed liquidity
    # (left end); max_cfar gives the HIGHEST (right end). Sweep floors between the
    # two so every point is a real, binding trade-off.
    lo = allocate(
        scen.npv_per_dollar, scen.liq_per_dollar,
        names=scen.names, caps=caps, budget=settings.capex_budget,
    ).cfar
    hi = max_cfar(
        scen.liq_per_dollar, caps=caps, budget=settings.capex_budget,
        alpha=settings.cfar_alpha,
    )
    floors = np.linspace(lo, 0.995 * hi, n_frontier)
    frontier = efficient_frontier(
        scen.npv_per_dollar, scen.liq_per_dollar,
        names=scen.names, caps=caps, budget=settings.capex_budget,
        cfar_floors=floors, alpha=settings.cfar_alpha,
    )

    logger.info(
        "Allocation: E[NPV]=$%.1fM  CFaR=$%.1fM  deployed=$%.0fM [%s]",
        alloc.expected_npv / 1e6, alloc.cfar / 1e6, alloc.deployed / 1e6, alloc.status,
    )
    return AllocationResult(
        allocation=alloc, frontier=frontier,
        budget=settings.capex_budget, cfar_floor=floor,
    )
