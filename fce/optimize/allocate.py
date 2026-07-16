"""CVXPY capital allocation with a Rockafellar–Uryasev CFaR constraint (Pillar 4).

Decision: how many dollars ``x_j`` to deploy into each project, subject to the
CapEx budget and each project's absorption cap. Objective: maximize expected
portfolio NPV. Guardrail: keep tail liquidity above a floor via a **convex
Cash-Flow-at-Risk constraint**.

**Locked decision #3:** continuous budget dollars (LP), open-source conic
solvers (CLARABEL/SCS/ECOS). No integers.

### Why CFaR (CVaR), not VaR
With discontinuous / heavy-tailed cash-flow distributions the VaR percentile
"may not be highly representative of the real risk that the portfolio owner
faces" (Fiorenzani et al., *The Handbook of Energy Trading*, 2012, p. 196) — a
rare tail event beyond the percentile can still sink liquidity. CVaR averages the
*whole* tail, and — crucially — is convex in the allocation.

### The Rockafellar–Uryasev linearization
CVaR at level ``alpha`` of a loss ``L(x)`` has the variational form
(Rockafellar & Uryasev, 2000, *Optimization of Conditional Value-at-Risk*):

    CVaR_alpha(L) = min_zeta  zeta + 1/((1-alpha)·S) · Σ_s [L_s(x) − zeta]_+

Introduce ``u_s ≥ L_s(x) − zeta``, ``u_s ≥ 0`` and it is linear. Here the loss is
negative portfolio liquidity, ``L_s = −(Q x)_s``; requiring ``CVaR_alpha(L) ≤
−floor`` guarantees the tail-average liquidity stays at or above ``floor``. That
tail-average guaranteed liquidity is what we report as **CFaR**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Allocation:
    """Solved allocation and its realized risk/return profile."""

    names: list[str]
    dollars: np.ndarray       # (N,) dollars deployed per project
    weights: np.ndarray       # (N,) fraction of the budget
    deployed: float           # total dollars deployed (<= budget)
    expected_npv: float       # E[portfolio NPV] ($)
    cfar: float               # tail-average liquidity at `alpha` ($) — the CFaR floor met
    status: str               # solver status
    portfolio_npv: np.ndarray # (S,) per-scenario portfolio NPV, for plotting


def allocate(
    npv_per_dollar: np.ndarray,
    liq_per_dollar: np.ndarray,
    *,
    names: list[str],
    caps: np.ndarray,
    budget: float,
    cfar_floor: float | None = None,
    alpha: float = 0.95,
    solver: str | None = None,
) -> Allocation:
    """Maximize expected portfolio NPV under budget, caps, and a CFaR floor.

    ``npv_per_dollar`` and ``liq_per_dollar`` are ``(S, N)`` scenario matrices
    (per dollar deployed). ``caps`` is ``(N,)`` max dollars each project absorbs.
    If ``cfar_floor`` is None the liquidity guardrail is dropped (pure
    NPV-maximizing baseline). Returns an :class:`Allocation`; check ``.status``.
    """
    import cvxpy as cp

    R = np.asarray(npv_per_dollar, float)
    Q = np.asarray(liq_per_dollar, float)
    s, n = R.shape

    # Normalize dollar quantities to $M inside the solve: raw magnitudes (~1e8)
    # across thousands of scenarios wreck conic-solver conditioning and produce
    # spurious "infeasible" statuses. Decision var x is in $M; recovered below.
    SCALE = 1e6
    budget_s = budget / SCALE
    caps_s = np.asarray(caps, float) / SCALE

    x = cp.Variable(n, nonneg=True)
    constraints = [cp.sum(x) <= budget_s, x <= caps_s]

    # Expected portfolio NPV ($M). R @ x is (S,).
    expected_npv = cp.sum(R @ x) / s
    objective = cp.Maximize(expected_npv)

    if cfar_floor is not None:
        # Rockafellar–Uryasev CVaR of the loss L = −(Q x), in $M.
        zeta = cp.Variable()
        u = cp.Variable(s, nonneg=True)
        loss = -(Q @ x)
        constraints += [u >= loss - zeta]
        cvar = zeta + cp.sum(u) / ((1.0 - alpha) * s)
        # tail-average liquidity ≥ floor  ⇔  CVaR(loss) ≤ −floor
        constraints += [cvar <= -cfar_floor / SCALE]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)

    if x.value is None:  # infeasible / failed — surface it, don't fabricate
        return Allocation(
            names=names, dollars=np.zeros(n), weights=np.zeros(n), deployed=0.0,
            expected_npv=float("nan"), cfar=float("nan"), status=prob.status,
            portfolio_npv=np.zeros(s),
        )

    dollars = np.asarray(x.value).clip(min=0.0) * SCALE  # back to dollars
    port_npv = R @ dollars
    port_liq = Q @ dollars
    # Realized CFaR (tail-average liquidity) of the solution, for reporting.
    var_cut = np.quantile(port_liq, 1.0 - alpha)
    tail = port_liq[port_liq <= var_cut]
    realized_cfar = float(tail.mean()) if tail.size else float(port_liq.min())

    return Allocation(
        names=names,
        dollars=dollars,
        weights=dollars / budget,
        deployed=float(dollars.sum()),
        expected_npv=float(port_npv.mean()),
        cfar=realized_cfar,
        status=prob.status,
        portfolio_npv=port_npv,
    )
