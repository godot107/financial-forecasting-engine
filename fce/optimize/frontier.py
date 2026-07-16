"""Risk–return efficient frontier: expected NPV vs. guaranteed liquidity (CFaR).

Re-solve the allocation across a sweep of CFaR floors. As the required tail
liquidity rises, the optimizer is forced out of the high-return but crude-exposed
projects, so expected NPV falls — tracing the trade-off curve a CFO actually
decides on. Optionally sweep budget tiers too (the card's $150M/$200M/$250M view).

This is the data behind the deck's "Risk-Return Frontier" slide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fce.optimize.allocate import Allocation, allocate


def max_cfar(
    liq_per_dollar: np.ndarray,
    *,
    caps: np.ndarray,
    budget: float,
    alpha: float = 0.95,
    solver: str | None = None,
) -> float:
    """Largest tail-liquidity (CFaR) the budget can guarantee — the frontier's
    safe endpoint. Maximizes ``t`` s.t. the Rockafellar–Uryasev CVaR of the loss
    stays at or below ``-t``. Returns dollars (``nan`` if the solve fails)."""
    import cvxpy as cp

    Q = np.asarray(liq_per_dollar, float)
    s, n = Q.shape
    SCALE = 1e6
    x = cp.Variable(n, nonneg=True)
    zeta = cp.Variable()
    u = cp.Variable(s, nonneg=True)
    t = cp.Variable()
    loss = -(Q @ x)
    cons = [
        cp.sum(x) <= budget / SCALE,
        x <= np.asarray(caps, float) / SCALE,
        u >= loss - zeta,
        zeta + cp.sum(u) / ((1.0 - alpha) * s) <= -t,
    ]
    prob = cp.Problem(cp.Maximize(t), cons)
    prob.solve(solver=solver)
    return float(t.value * SCALE) if t.value is not None else float("nan")


@dataclass
class FrontierPoint:
    budget: float
    cfar_floor: float
    expected_npv: float
    cfar: float
    deployed: float
    weights: np.ndarray
    status: str


def efficient_frontier(
    npv_per_dollar: np.ndarray,
    liq_per_dollar: np.ndarray,
    *,
    names: list[str],
    caps: np.ndarray,
    budget: float,
    cfar_floors: np.ndarray,
    alpha: float = 0.95,
    solver: str | None = None,
) -> list[FrontierPoint]:
    """Trace the NPV↔CFaR frontier over ``cfar_floors`` at a fixed ``budget``.

    Infeasible floors (too demanding for the available projects) are recorded
    with their status and skipped by plotters via ``np.isnan(expected_npv)``.
    """
    points: list[FrontierPoint] = []
    for floor in np.asarray(cfar_floors, float):
        alloc: Allocation = allocate(
            npv_per_dollar, liq_per_dollar, names=names, caps=caps,
            budget=budget, cfar_floor=float(floor), alpha=alpha, solver=solver,
        )
        points.append(
            FrontierPoint(
                budget=budget, cfar_floor=float(floor),
                expected_npv=alloc.expected_npv, cfar=alloc.cfar,
                deployed=alloc.deployed, weights=alloc.weights, status=alloc.status,
            )
        )
    return points
