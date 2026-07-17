"""What-if / risk-analysis module (feeds Pillar 4 risk + Pillar 5 deck).

A thin layer over the accounting + Monte Carlo machinery, in three flavors:

1. **Named deterministic scenarios** — e.g. *SOFR +200bp & WTI −30% & CPI +4%*
   run through the deterministic accounting layer (Berk & DeMarzo, *Corporate
   Finance*, capital-budgeting scenario analysis).
2. **One-at-a-time sensitivity → tornado** on NPV / CFaR. Complements (not
   duplicates) the SHAP item: SHAP explains the *driver model*; the tornado
   attributes *accounting-output* movement.
3. **Reverse stress test** — solve for the driver combination that breaches the
   covenant / pushes CFaR below the liquidity floor (optimize over driver space;
   reuse CVXPY). Answers "what breaks us?" — the most compelling deck slide.

Doubles as **SR 11-7 model-governance evidence** (Edwards, *Energy Trading &
Investing*, p. 464): sensitivity/stress testing demonstrates the model is
conservative.

**Causally-correct semantics (bounded scope):** scenarios are applied as Pearl
``do()`` interventions over a *hand-specified* macro DAG (:mod:`fce.scenarios.scm`),
NOT observational conditioning — correlated drivers make naive conditioning the
wrong counterfactual. **SCOPE GUARD — do NOT build** causal discovery
(PC/GES/NOTEARS) or treatment-effect estimation (ATE/DiD/IV): shaky identifiability,
synthetic entities have no real treatment, and it overlaps the dedicated
retail-pricing causal project. Keep this project's identity = probabilistic
simulation + optimization.
"""

from fce.scenarios.scm import (
    MacroSample,
    MacroSCM,
    combine,
    commodity_shock,
    rate_shock,
)
from fce.scenarios.whatif import (
    DEFAULT_SCENARIOS,
    ReverseStress,
    ScenarioContext,
    ScenarioResult,
    TornadoBar,
    build_context,
    reverse_stress,
    run_scenarios,
    tornado,
)

__all__ = [
    "MacroSCM", "MacroSample", "rate_shock", "commodity_shock", "combine",
    "build_context", "ScenarioContext",
    "run_scenarios", "ScenarioResult", "DEFAULT_SCENARIOS",
    "tornado", "TornadoBar",
    "reverse_stress", "ReverseStress",
]
