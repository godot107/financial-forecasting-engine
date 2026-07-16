"""Tests for the CVXPY allocator and the CFaR (Rockafellar–Uryasev) constraint.

Uses small hand-built scenario matrices (not the full sim) so the tests are fast
and the economics are controlled: project 0 = high return / fat tail, project 1 =
modest return / safe tail. The CFaR constraint must rotate the book from 0 → 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from fce.optimize import allocate, efficient_frontier, max_cfar


def _scenarios(seed=0, s=2000):
    rng = np.random.default_rng(seed)
    # Project 0: higher mean NPV/$, but wide + low-tail liquidity.
    r0 = rng.normal(0.30, 0.40, s)
    q0 = rng.normal(1.20, 0.55, s)
    # Project 1: lower mean NPV/$, but tight + high-tail (safe) liquidity.
    r1 = rng.normal(0.15, 0.05, s)
    q1 = rng.normal(1.10, 0.05, s)
    R = np.column_stack([r0, r1])
    Q = np.column_stack([q0, q1])
    return R, Q


NAMES = ["risky", "safe"]
CAPS = np.array([1e12, 1e12])   # non-binding by default
BUDGET = 100e6


def test_respects_budget_and_caps():
    R, Q = _scenarios()
    caps = np.array([30e6, 1e12])
    a = allocate(R, Q, names=NAMES, caps=caps, budget=BUDGET)
    assert a.status == "optimal"
    assert a.deployed <= BUDGET + 1.0
    assert a.dollars[0] <= 30e6 + 1.0            # cap on project 0 binds
    assert np.all(a.dollars >= -1e-6)


def test_unconstrained_prefers_higher_mean():
    R, Q = _scenarios()
    a = allocate(R, Q, names=NAMES, caps=CAPS, budget=BUDGET)
    # Positive expected returns → deploy the whole budget into the best project.
    assert a.deployed == pytest.approx(BUDGET, rel=1e-3)
    assert a.dollars[0] > a.dollars[1]


def test_cfar_constraint_rotates_toward_safe_and_is_met():
    R, Q = _scenarios()
    base = allocate(R, Q, names=NAMES, caps=CAPS, budget=BUDGET)
    # Demand more guaranteed liquidity than the unconstrained book provides.
    floor = base.cfar * 1.05
    con = allocate(R, Q, names=NAMES, caps=CAPS, budget=BUDGET, cfar_floor=floor)
    assert con.status == "optimal"
    # Book rotates toward the safe project...
    assert con.weights[1] > base.weights[1]
    # ...the floor is actually met (realized CFaR within solver tolerance)...
    assert con.cfar >= floor * 0.99
    # ...and it costs expected NPV (no free lunch).
    assert con.expected_npv <= base.expected_npv + 1e-6


def test_infeasible_floor_is_reported_not_faked():
    R, Q = _scenarios()
    # A floor far above the max achievable tail liquidity is infeasible.
    a = allocate(R, Q, names=NAMES, caps=CAPS, budget=BUDGET, cfar_floor=1e12)
    assert a.status in ("infeasible", "infeasible_inaccurate")
    assert np.isnan(a.expected_npv)


def test_frontier_slopes_down_between_real_endpoints():
    R, Q = _scenarios()
    lo = allocate(R, Q, names=NAMES, caps=CAPS, budget=BUDGET).cfar   # max-NPV end
    hi = max_cfar(Q, caps=CAPS, budget=BUDGET)                        # max-safety end
    floors = np.linspace(lo, 0.99 * hi, 8)
    pts = [p for p in efficient_frontier(R, Q, names=NAMES, caps=CAPS, budget=BUDGET, cfar_floors=floors)
           if not np.isnan(p.expected_npv)]
    npvs = [p.expected_npv for p in pts]
    # Raising the liquidity floor never increases achievable NPV (tol = solver noise).
    tol = 1e-4 * abs(npvs[0])
    assert all(npvs[i] >= npvs[i + 1] - tol for i in range(len(npvs) - 1))
    # And the trade-off is real: the safe end gives up NPV vs. the max-NPV end.
    assert npvs[-1] < npvs[0]
