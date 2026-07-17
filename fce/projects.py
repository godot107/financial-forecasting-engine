"""The five synthetic capital projects (Alpha–Epsilon) and their scenario sim.

These are the competing CapEx initiatives the optimizer allocates across. They
are **synthetic** (locked decision — no realized history), but their per-scenario
NPV and liquidity distributions are *genuinely computed* by pushing a shared
macro driver through the deterministic accounting engine (:mod:`fce.accounting`),
so the optimizer's recommendation is a real optimization, not a hard-coded number.

Design so diversification actually matters:

* One **systematic** driver (WTI) hits every project's revenue, with a
  project-specific **beta** (LNG/chemical ride crude hard; battery/clean are
  nearly price-insensitive and lean on volatility instead).
* Each project also carries **idiosyncratic** revenue noise, so the projects are
  correlated but not identical — the optimizer can trade expected NPV against
  tail liquidity by mixing high-beta and low-beta projects.

Until Pillar 1 (NumPyro) lands, the WTI paths are a placeholder GBM; swap
:func:`_driver_paths` for the real driver and everything downstream is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fce.accounting import build_statements, npv, present_value
from fce.config import Settings


@dataclass(frozen=True)
class Project:
    """One synthetic CapEx initiative. ``invested`` is the reference capital ($)."""

    name: str
    invested: float          # reference capital deployed ($), = opening PP&E
    wti_beta: float          # revenue sensitivity to WTI (0 = price-insensitive)
    idio_vol: float          # idiosyncratic monthly revenue vol
    base_revenue: float      # $/month revenue at the reference WTI
    opex_ratio: float        # opex as a fraction of revenue
    debt: float              # opening debt ($) -> interest drag
    credit_spread: float = 0.02  # over the risk-free short rate on floating debt


# Five competing initiatives with deliberately different risk/return shapes.
# ``wti_beta`` spans a genuine hedge (Gamma < 0: grid storage earns MORE when
# crude is weak/volatile) to heavy crude exposure (Beta LNG), so the optimizer
# has a real diversification lever. ``idio_vol`` is a PERSISTENT per-scenario
# factor (see :func:`simulate_project_scenarios`), which keeps cross-project
# correlation well below 1 instead of washing out over the horizon.
PROJECTS: list[Project] = [
    Project("Alpha (Clean Retrofit)",   invested=55e6,  wti_beta=0.10,  idio_vol=0.14, base_revenue=10.2e6, opex_ratio=0.58, debt=20e6, credit_spread=0.018),
    Project("Beta (LNG Terminal)",      invested=90e6,  wti_beta=1.15,  idio_vol=0.34, base_revenue=18.6e6, opex_ratio=0.60, debt=60e6, credit_spread=0.030),
    Project("Gamma (Grid Battery)",     invested=40e6,  wti_beta=-0.35, idio_vol=0.08, base_revenue=5.6e6,  opex_ratio=0.54, debt=12e6, credit_spread=0.015),
    Project("Delta (Pipeline IoT)",     invested=30e6,  wti_beta=0.30,  idio_vol=0.08, base_revenue=4.7e6,  opex_ratio=0.56, debt=8e6,  credit_spread=0.016),
    Project("Epsilon (Chemical Line)",  invested=70e6,  wti_beta=0.85,  idio_vol=0.28, base_revenue=15.0e6, opex_ratio=0.62, debt=45e6, credit_spread=0.028),
]


def _driver_paths(settings: Settings) -> np.ndarray:
    """Shared systematic WTI paths, ``(S, T)``. Placeholder GBM (→ Pillar 1)."""
    rng = np.random.default_rng(settings.seed)
    s, t = settings.n_paths, settings.horizon_months
    mu, sigma, s0 = 0.0, 0.08, 75.0
    shocks = rng.normal(mu - 0.5 * sigma**2, sigma, size=(s, t))
    return s0 * np.exp(np.cumsum(shocks, axis=1)), s0


@dataclass
class ProjectScenarios:
    """Per-scenario, per-dollar metrics ready for the optimizer.

    ``npv_per_dollar`` and ``liq_per_dollar`` are ``(S, N)``; row ``s`` is one
    Monte-Carlo state, column ``j`` is project ``j``. "Per dollar" = the metric
    divided by that project's reference ``invested`` capital, so a dollar
    allocation ``x_j`` contributes ``npv_per_dollar[:, j] * x_j`` to portfolio NPV.
    """

    names: list[str]
    npv_per_dollar: np.ndarray
    liq_per_dollar: np.ndarray


def simulate_project_scenarios(
    settings: Settings | None = None,
    projects: list[Project] | None = None,
    *,
    wti_paths: np.ndarray | None = None,
    wti0: float | None = None,
    short_rates: np.ndarray | None = None,
    demand_paths: np.ndarray | None = None,
) -> ProjectScenarios:
    """Run every project through the accounting engine → ``(S, N)`` metric matrices.

    For each project: WTI (systematic) + idiosyncratic noise → revenue → balanced
    3-statement articulation → discounted NPV and total (undiscounted) FCFF
    liquidity. Normalized by reference capital so the results are per-dollar.

    ``wti_paths`` (``(S, T)``) injects an external driver — e.g. the Pillar-1 HMM
    posterior-predictive paths from :func:`fce.drivers.wti_scenario_paths`; pass
    its ``wti0`` too. When omitted, the fast placeholder GBM driver is used.

    ``short_rates`` (``(S, T)``) injects Pillar-2 rate paths (from
    :func:`fce.term_structure.rate_scenario_paths`): each project's debt service
    becomes **floating-rate** and NPV is discounted with **pathwise** factors from
    the curve. When omitted, a fixed coupon and flat WACC are used.

    ``demand_paths`` (``(S, T)``, a multiplier centered on 1) injects a macro
    demand driver — e.g. the ``demand`` node of the Pillar-scenarios
    :class:`~fce.scenarios.scm.MacroSCM`, so a rate/inflation shock flows into
    revenue. When omitted, demand is neutral (1.0).
    """
    settings = settings or Settings()
    projects = projects or PROJECTS
    if wti_paths is None:
        wti, wti0 = _driver_paths(settings)          # placeholder GBM
    else:
        wti = np.asarray(wti_paths, dtype=float)
        if wti0 is None:
            wti0 = float(wti[:, 0].mean())
    s, t = wti.shape
    wti_ret = wti / wti0  # systematic multiplier, mean ~1

    # Pillar-2 stochastic discounting (per-path DFs) when rate paths are supplied.
    discount_fac = None
    if short_rates is not None:
        from fce.term_structure import discount_factors_from_short_rates

        short_rates = np.asarray(short_rates, dtype=float)
        if short_rates.shape != (s, t):
            raise ValueError(
                f"short_rates shape {short_rates.shape} must match drivers {(s, t)}"
            )
        discount_fac = discount_factors_from_short_rates(
            short_rates, spread=settings.wacc_spread
        )

    rng = np.random.default_rng(settings.seed + 1)
    npv_cols, liq_cols, names = [], [], []

    for j, proj in enumerate(projects):
        # Revenue = base × (WTI multiplier ** beta) × idiosyncratic noise, where
        # the idio factor is PERSISTENT per scenario (one draw held across the
        # horizon) so it survives the sum-over-T instead of averaging away — this
        # is what keeps the projects imperfectly correlated. A small monthly
        # wiggle is layered on top.
        idio_scenario = np.exp(
            rng.normal(-0.5 * proj.idio_vol**2, proj.idio_vol, size=(s, 1))
        )
        idio_month = np.exp(rng.normal(-0.5 * 0.03**2, 0.03, size=(s, t)))
        revenue = proj.base_revenue * (wti_ret ** proj.wti_beta) * idio_scenario * idio_month
        if demand_paths is not None:
            revenue = revenue * np.clip(demand_paths, 0.0, None)  # macro demand multiplier

        # Debt service: floating-rate off the simulated curve (Pillar 2) if rate
        # paths are supplied, else a fixed ~6% annual coupon.
        if short_rates is not None:
            from fce.term_structure import floating_rate_interest

            interest = floating_rate_interest(
                short_rates, notional=proj.debt, credit_spread=proj.credit_spread,
            )
        else:
            interest = np.full((s, t), proj.debt * 0.06 / 12)

        capex = np.full((s, t), proj.invested / t * 0.30)  # sustaining capex
        stmt = build_statements(
            revenue=revenue,
            opex=proj.opex_ratio * revenue,
            depreciation=capex,                 # steady-state: dep ≈ sustaining capex
            capex=capex,
            delta_nwc=0.02 * revenue,
            interest=interest,
            net_debt_issued=np.zeros((s, t)),
            tax_rate=settings.tax_rate,
            opening_ppe=proj.invested,
            opening_debt=proj.debt,
        )

        # Capital-budgeting NPV nets the upfront investment: NPV = −C0 + PV(FCFF).
        # Discount with pathwise term-structure factors (Pillar 2) or flat WACC.
        if discount_fac is not None:
            pv = present_value(stmt.fcff, discount_fac)
        else:
            pv = npv(stmt.fcff, settings.wacc)
        project_npv = pv - proj.invested   # (S,)
        liquidity = stmt.fcff.sum(axis=1)                             # (S,) cash thrown off
        npv_cols.append(project_npv / proj.invested)
        liq_cols.append(liquidity / proj.invested)
        names.append(proj.name)

    return ProjectScenarios(
        names=names,
        npv_per_dollar=np.column_stack(npv_cols),
        liq_per_dollar=np.column_stack(liq_cols),
    )
