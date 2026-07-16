# Probabilistic Financial Forecasting & Hybrid Accounting Engine

A production-grade quantitative engine for allocating a multi-year CapEx budget
across competing energy/industrial projects **under macroeconomic uncertainty** —
bridging probabilistic ML, quantitative finance, deterministic accounting, and
stochastic optimization.

> The architectural bet: keep accounting identities and discounting **100 %
> deterministic and auditable** (GAAP/IFRS-compliant, zero drift), while driving
> their inputs dynamically via Bayesian simulation. Spreadsheets give single-point
> estimates that hide tail risk; this engine quantifies **Cash-Flow-at-Risk
> (CFaR)** across 10,000+ simulated market states.

## The four pillars

1. **Probabilistic ML drivers** (`NumPyro`) — Bayesian HMM/state-space models of
   rates, inflation, and commodity prices → thousands of Monte Carlo trajectories
   with honest P10/P50/P90 intervals.
2. **Term structure & debt** (`QuantLib`) — bootstrap yield curves, price
   floating-rate debt and hedges with exact day-count conventions.
3. **Deterministic accounting** (`NumPy`/`JAX`) — fully balanced 3-statement
   articulation (Revenue → EBITDA → FCFF → NPV), vectorized across all paths,
   guarded by golden identity tests.
4. **Stochastic optimization** (`CVXPY`) — convex capital budgeting that maximizes
   expected portfolio NPV subject to the budget ceiling and a CFaR liquidity floor.

Validation: purged + embargoed rolling-origin backtesting (López de Prado, AFML
Ch. 7), interval coverage / pinball loss, and policy replay vs. naive baselines
across historical stress windows (2020, 2022). Culminates in a mock CFO decision
deck: efficient frontier, "Tails vs. Spreadsheet," and SR 11-7 model-governance
evidence.

## Status

Scaffolded and MVP-runnable. The **deterministic accounting core is built and
tested**; a placeholder driver wires the full slice end-to-end today. The
probabilistic (Pillar 1), term-structure (Pillar 2), and optimization (Pillar 4)
engines are stubbed with defined interfaces. See `CLAUDE.md` for architecture,
locked decisions, and build order.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m fce      # MVP capital-forecasting slice
pytest             # golden accounting-identity + leakage-free split tests
```

Data ingestion uses the free EIA and FRED APIs — copy `.env.example` to `.env`
and add your keys.
