"""Pillar 4 — stochastic capital allocation under Cash-Flow-at-Risk (CVXPY).

Ingests the simulated multi-year FCFF/NPV distributions across competing
projects and solves for the CapEx allocation that maximizes expected portfolio
NPV subject to the annual budget ceiling and a CFaR liquidity floor.

**Locked decision #3: continuous convex allocation (budget fractions, LP/QP)**
on open-source solvers (ECOS/SCS). Binary go/no-go (MIP, needs CBC/SCIP) is an
optional later extension where CVXPY gets fragile.

CFaR is encoded as a **convex constraint via the Rockafellar–Uryasev CVaR
linearization** over the scenario matrix, so the whole program stays convex.
CVaR/CFaR is preferred over VaR because with heavy-tailed cash flows the VaR
percentile "may not be highly representative of the real risk" (Handbook of
Energy Trading, p. 196) — exactly the tail-breach case this project sells.

Implemented: :func:`allocate` (single solve) and :func:`efficient_frontier`
(NPV↔CFaR sweep). Scenario inputs come from :func:`fce.projects.simulate_project_scenarios`.
"""

from fce.optimize.allocate import Allocation, allocate, evaluate
from fce.optimize.frontier import FrontierPoint, efficient_frontier, max_cfar

__all__ = [
    "Allocation", "allocate", "evaluate",
    "FrontierPoint", "efficient_frontier", "max_cfar",
]
