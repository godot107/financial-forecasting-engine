"""Policy backtest — replay the allocation over history vs. naive baselines.

Distinct from the statistical loop: given a *realized* driver path, the allocation
decision is deterministic, so there is no pinball loss here. We compare realized
cumulative portfolio NPV and cash-reserve drawdowns of the CVXPY stochastic
allocation against baseline heuristics (naive IRR ranking, equal weighting) across
historical stress windows (2020 pandemic, 2022 rate/inflation shock).

STUB — implement once Pillar 4 (optimize) lands. See card Phase 4.
"""

from __future__ import annotations

from typing import Callable


def replay_policy(*args, **kwargs):
    """Replay an allocation ``policy`` over historical driver realizations.

    Intended signature (to be finalized): ``replay_policy(drivers, projects,
    policy: Callable, baselines: dict[str, Callable]) -> PolicyReport``.
    """
    raise NotImplementedError(
        "Policy replay is a Phase-4 deliverable — see fce/optimize/allocate.py first."
    )
