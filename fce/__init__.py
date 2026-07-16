"""Probabilistic Financial Forecasting & Hybrid Accounting Engine (``fce``).

Four-pillar hybrid capital-allocation engine:

1. ``drivers``         — NumPyro probabilistic macro drivers (HMM / state-space).
2. ``term_structure``  — QuantLib yield curves + debt servicing.
3. ``accounting``      — deterministic, vectorized 3-statement engine (Revenue→FCFF→NPV).
4. ``optimize``        — CVXPY capital allocation under Cash-Flow-at-Risk (CFaR).

The design invariant: **the accounting and discounting math stays 100 %
deterministic and auditable**; all uncertainty enters *upstream* as simulated
driver trajectories. See ``CLAUDE.md`` for the locked pre-build decisions.
"""

__version__ = "0.0.1"
