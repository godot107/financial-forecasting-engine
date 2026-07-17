"""What-if / stress-analysis tools over the macro SCM + accounting engine.

Three flavors, all evaluating a **locked** allocation (the recommended policy)
under ``do()``-intervened drivers — so they answer "what happens to *our plan* if
the world shifts," which is the CFO's question:

1. :func:`run_scenarios` — named deterministic scenarios (e.g. SOFR +200bp,
   WTI −30%, both together).
2. :func:`tornado` — one-at-a-time ±shock per driver, ranked by NPV swing.
3. :func:`reverse_stress` — grid the driver-shock space and find where CFaR
   breaches the liquidity floor ("what breaks us?").

Scenarios are applied as interventions over :class:`~fce.scenarios.scm.MacroSCM`,
**not** observational conditioning — see that module for why. Doubles as SR 11-7
model-governance evidence (sensitivity/stress testing demonstrates conservatism).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fce.config import Settings, get_settings
from fce.optimize import allocate, evaluate
from fce.projects import PROJECTS, _driver_paths, simulate_project_scenarios
from fce.scenarios.scm import (
    MacroSample,
    MacroSCM,
    combine,
    commodity_shock,
    rate_shock,
)


# --------------------------------------------------------------------------- #
# Baseline plumbing
# --------------------------------------------------------------------------- #
def baseline_drivers(settings: Settings, *, use_hmm: bool = False):
    """Exogenous SCM roots: ``(wti_paths, wti0, short_rates)``.

    Fast by default (placeholder GBM + Vasicek); ``use_hmm`` swaps in the fitted
    NumPyro driver. Rates come from the bootstrapped-curve Vasicek simulation.
    """
    from fce.term_structure import rate_scenario_paths

    if use_hmm:
        from fce.drivers import wti_scenario_paths

        wti, wti0 = wti_scenario_paths(settings)
    else:
        wti, wti0 = _driver_paths(settings)
    short_rates, _ = rate_scenario_paths(settings)
    return wti, wti0, short_rates


def _matrices(settings, sample: MacroSample, wti0: float):
    """Project scenario matrices for a (possibly intervened) macro sample."""
    return simulate_project_scenarios(
        settings, wti_paths=sample.commodity, wti0=wti0,
        short_rates=sample.rates, demand_paths=sample.demand,
    )


@dataclass
class ScenarioContext:
    """Everything a stress tool needs: locked policy + baseline world."""

    settings: Settings
    scm: MacroSCM
    base_sample: MacroSample
    wti0: float
    dollars: np.ndarray            # the locked allocation ($ per project)
    base_npv: float
    base_cfar: float
    scen: object = None            # baseline ProjectScenarios (for the frontier/report)
    caps: np.ndarray = None        # per-project absorption caps
    names: list = None             # project names


def build_context(settings: Settings | None = None, *, use_hmm: bool = False) -> ScenarioContext:
    """Simulate the baseline world and lock in the recommended allocation."""
    settings = settings or get_settings()
    scm = MacroSCM()
    wti, wti0, rates = baseline_drivers(settings, use_hmm=use_hmm)
    base_sample = scm.simulate(rates, wti, seed=settings.seed)

    scen = _matrices(settings, base_sample, wti0)
    caps = np.array([p.invested for p in PROJECTS])
    alloc = allocate(
        scen.npv_per_dollar, scen.liq_per_dollar, names=scen.names, caps=caps,
        budget=settings.capex_budget, cfar_floor=settings.cfar_floor,
        alpha=settings.cfar_alpha,
    )
    npv, cfar = evaluate(scen.npv_per_dollar, scen.liq_per_dollar, alloc.dollars,
                         alpha=settings.cfar_alpha)
    return ScenarioContext(
        settings, scm, base_sample, wti0, alloc.dollars, npv, cfar,
        scen=scen, caps=caps, names=scen.names,
    )


def _score(ctx: ScenarioContext, interventions: dict) -> tuple[float, float]:
    """(E[NPV], CFaR) of the locked policy under ``do(interventions)``."""
    sample = ctx.scm.do(ctx.base_sample, interventions) if interventions else ctx.base_sample
    scen = _matrices(ctx.settings, sample, ctx.wti0)
    return evaluate(scen.npv_per_dollar, scen.liq_per_dollar, ctx.dollars,
                    alpha=ctx.settings.cfar_alpha)


# --------------------------------------------------------------------------- #
# 1 · named scenarios
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioResult:
    name: str
    expected_npv: float
    cfar: float
    d_npv: float
    d_cfar: float
    breaches_floor: bool


# The canonical deck scenarios.
DEFAULT_SCENARIOS: dict[str, dict] = {
    "SOFR +200bp": rate_shock(200),
    "WTI −30%": commodity_shock(-30),
    "Rate spike + oil crash": combine(rate_shock(200), commodity_shock(-30)),
    "Soft landing (SOFR −100bp)": rate_shock(-100),
}


def run_scenarios(
    settings: Settings | None = None,
    *,
    scenarios: dict[str, dict] | None = None,
    ctx: ScenarioContext | None = None,
) -> tuple[list[ScenarioResult], ScenarioContext]:
    """Score the locked policy under each named scenario. Returns (results, ctx)."""
    ctx = ctx or build_context(settings)
    scenarios = scenarios if scenarios is not None else DEFAULT_SCENARIOS
    floor = ctx.settings.cfar_floor

    results = [ScenarioResult("Baseline", ctx.base_npv, ctx.base_cfar, 0.0, 0.0,
                              ctx.base_cfar < floor)]
    for name, iv in scenarios.items():
        npv, cfar = _score(ctx, iv)
        results.append(ScenarioResult(
            name, npv, cfar, npv - ctx.base_npv, cfar - ctx.base_cfar, cfar < floor,
        ))
    return results, ctx


# --------------------------------------------------------------------------- #
# 2 · tornado sensitivity
# --------------------------------------------------------------------------- #
@dataclass
class TornadoBar:
    driver: str
    npv_low: float     # NPV under the adverse move
    npv_high: float    # NPV under the favorable move
    swing: float       # |high − low|


def tornado(
    ctx: ScenarioContext | None = None,
    *,
    settings: Settings | None = None,
    rate_bp: float = 100.0,
    wti_pct: float = 20.0,
) -> tuple[list[TornadoBar], ScenarioContext]:
    """One-at-a-time ±shock per driver → NPV swing, sorted largest first.

    Attributes movement in the *accounting output* — complementary to a SHAP
    explanation of the driver model (different question, both worth showing).
    """
    ctx = ctx or build_context(settings)
    specs = {
        f"Rates ±{rate_bp:.0f}bp": (rate_shock(rate_bp), rate_shock(-rate_bp)),
        f"WTI ±{wti_pct:.0f}%": (commodity_shock(-wti_pct), commodity_shock(wti_pct)),
    }
    bars = []
    for driver, (adverse, favorable) in specs.items():
        npv_low, _ = _score(ctx, adverse)
        npv_high, _ = _score(ctx, favorable)
        bars.append(TornadoBar(driver, npv_low, npv_high, abs(npv_high - npv_low)))
    bars.sort(key=lambda b: b.swing, reverse=True)
    return bars, ctx


# --------------------------------------------------------------------------- #
# 3 · reverse stress test
# --------------------------------------------------------------------------- #
@dataclass
class ReverseStress:
    rate_bp: np.ndarray        # (nr,) rate-shock axis (basis points)
    wti_pct: np.ndarray        # (nw,) WTI-shock axis (%)
    cfar: np.ndarray           # (nr, nw) CFaR of the locked policy
    floor: float
    base_cfar: float

    def breaches(self) -> np.ndarray:
        """Boolean mask of (rate, wti) combos that push CFaR below the floor."""
        return self.cfar < self.floor


def reverse_stress(
    ctx: ScenarioContext | None = None,
    *,
    settings: Settings | None = None,
    rate_bp_max: float = 400.0,
    wti_pct_max: float = 50.0,
    covenant_floor: float | None = None,
    n: int = 13,
) -> tuple[ReverseStress, ScenarioContext]:
    """Grid the (rate-hike, oil-crash) space; find where CFaR breaches a covenant.

    Answers "what combination of shocks breaks our liquidity?" — the dangerous
    shocks are the ones *not* in the historical sample, so we search for them.

    The breach threshold is a **hard covenant floor**, distinct from (and below)
    the CFaR *target* the allocation is optimized to — the recommended policy sits
    at that target by construction, so a covenant with headroom is what yields a
    meaningful breach frontier. Defaults to 85% of the baseline CFaR.
    """
    ctx = ctx or build_context(settings)
    floor = covenant_floor if covenant_floor is not None else 0.85 * ctx.base_cfar
    rate_axis = np.linspace(0.0, rate_bp_max, n)
    wti_axis = np.linspace(0.0, -wti_pct_max, n)
    grid = np.empty((n, n))
    for i, rb in enumerate(rate_axis):
        for j, wp in enumerate(wti_axis):
            _, cfar = _score(ctx, combine(rate_shock(rb), commodity_shock(wp)))
            grid[i, j] = cfar
    rs = ReverseStress(rate_axis, wti_axis, grid, floor, ctx.base_cfar)
    return rs, ctx
