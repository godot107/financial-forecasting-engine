# Probabilistic Financial Forecasting & Hybrid Accounting Engine

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-29%20passing-brightgreen)
![Stack](https://img.shields.io/badge/stack-NumPyro%20%C2%B7%20JAX%20%C2%B7%20CVXPY%20%C2%B7%20QuantLib-8A2BE2)

A quantitative engine for allocating a multi-year CapEx budget across competing
energy/industrial projects **under macroeconomic uncertainty** — bridging
probabilistic ML, quantitative finance, deterministic accounting, and stochastic
optimization.

> **The architectural bet:** keep the accounting identities and discounting **100 %
> deterministic and auditable** (GAAP-style, zero drift), while driving their
> inputs with Bayesian simulation. A spreadsheet gives one NPV and hides the tail;
> this engine quantifies **Cash-Flow-at-Risk (CFaR)** across 10,000+ simulated
> market states and optimizes the allocation against it.

<p align="center">
  <img src="notebooks/figures/02_frontier.png" width="80%" alt="Risk–return efficient frontier: expected NPV vs. guaranteed liquidity (CFaR)">
</p>

*The decision a capital committee actually makes: how much expected NPV to trade
for how much guaranteed liquidity. Every point is a re-solved convex allocation.*

---

## Why

Traditional corporate-finance models are static spreadsheets: single-point
assumptions in, a single NPV out. They break when simultaneous shocks — an oil
spike, a rate hike, cost inflation — interact and threaten debt covenants. This
engine keeps the parts that *must* be exact (accounting identities, discounting)
deterministic and auditable, and injects uncertainty only where it belongs: in
the macro drivers.

## The four pillars

| # | Pillar | Tech | Status |
|---|--------|------|--------|
| 1 | **Probabilistic drivers** — regime-switching HMM of commodity prices → Monte-Carlo trajectories with honest P10/P50/P90 | NumPyro / JAX | ✅ built |
| 2 | **Term structure & debt** — bootstrap yield curves, Vasicek short rates, floating-rate debt service | QuantLib | ✅ built |
| 3 | **Deterministic accounting** — balanced 3-statement articulation (Revenue → FCFF → NPV), vectorized, golden-tested | NumPy / JAX | ✅ built |
| 4 | **Stochastic allocation** — convex capital budgeting under a CFaR floor | CVXPY | ✅ built |

Validation: purged + embargoed rolling-origin backtesting (López de Prado, *AFML*
Ch. 7), interval coverage / pinball loss, and policy replay across historical
stress windows — culminating in a mock CFO decision deck.

## What it demonstrates

**Accounting that balances by construction — and proves it.** The 3-statement
engine is a vectorized roll-forward where ΔAssets ≡ ΔLiabilities + ΔEquity every
period; golden identity tests (`assert_balanced`) run in CI *and* inside the
pipeline on every run.

**A driver that models regimes, not a random walk.** A Bayesian Gaussian HMM fit
with NUTS (discrete states marginalized via the forward algorithm) captures calm
vs. crisis volatility — and puts materially more mass in the tails than the
matched GBM it replaces.

<p align="center">
  <img src="notebooks/figures/05_regimes.png" width="49%" alt="HMM-decoded WTI regimes">
  <img src="notebooks/figures/04_tails_vs_spreadsheet.png" width="49%" alt="Tails vs. Spreadsheet">
</p>

**Risk-aware optimization.** Capital is allocated by a convex program that
maximizes expected portfolio NPV subject to the budget, per-project caps, and a
**Cash-Flow-at-Risk floor** encoded via the Rockafellar–Uryasev CVaR
linearization — so the recommendation and the efficient frontier are *computed*,
not asserted.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'          # core + tests (fast; no heavy pillars)
pip install -r requirements.txt  # full 4-pillar stack (NumPyro, CVXPY, QuantLib, …)

python -m fce                    # MVP accounting slice (Pillar 3)
python -m fce --allocate         # CFaR-constrained allocation (Pillars 3+4)
python -m fce --allocate --hmm   # drive it with the NumPyro HMM (Pillar 1)
python -m fce --allocate --hmm --quantlib   # + term-structure discounting & floating debt (Pillar 2)
pytest                           # 29 tests: golden identities, leakage-free splits, model recovery, curve/debt
```

No API keys required to run — the driver falls back to a reproducible synthetic
WTI series offline. For real data, copy `.env.example` to `.env` and add free
[EIA](https://www.eia.gov/opendata/) and [FRED](https://fred.stlouisfed.org/docs/api/)
keys.

## Notebooks

Each pillar ships an executable tutorial notebook whose claims are grounded in
canonical references (Berk & DeMarzo, Hilpisch, Dixon et al., López de Prado,
Rockafellar & Uryasev):

- [`01_deterministic_accounting_core.ipynb`](notebooks/01_deterministic_accounting_core.ipynb) — FCFF, balances-by-construction, golden identities, Monte-Carlo NPV.
- [`02_capital_allocation_cfar.ipynb`](notebooks/02_capital_allocation_cfar.ipynb) — risk-return frontier, recommended allocation, "Tails vs. Spreadsheet" *(Plotly)*.
- [`03_probabilistic_drivers_hmm.ipynb`](notebooks/03_probabilistic_drivers_hmm.ipynb) — regime decoding, posterior-predictive fan, HMM vs. GBM tails *(Plotly)*.
- [`04_term_structure_quantlib.ipynb`](notebooks/04_term_structure_quantlib.ipynb) — bootstrapped curve, Vasicek short-rate fan, floating-debt sensitivity *(Plotly)*.

## Project layout

```
fce/
  ingest/         # EIA + FRED clients, pinned-vintage parquet cache
  drivers/        # Pillar 1 — NumPyro regime-switching HMM
  term_structure/ # Pillar 2 — QuantLib curves + debt   (stub)
  accounting/     # Pillar 3 — deterministic 3-statement engine + golden tests
  optimize/       # Pillar 4 — CVXPY allocation + Rockafellar–Uryasev CFaR
  scenarios/      # what-if / macro-SCM interventions    (stub)
  backtest/       # purged + embargoed CV, coverage/pinball
  projects.py     # the 5 synthetic projects → scenario simulation
  pipeline.py     # orchestrator-agnostic entrypoints
notebooks/        # tutorials + exported deck figures
tests/            # golden identities, split leakage, optimizer, model recovery
```

## Roadmap

- **Scenarios / SCM:** `do()`-intervention what-ifs (SOFR +200bp, WTI −30%) and a
  reverse stress test over a hand-specified macro DAG.
- **Executive deck:** efficient frontier, tails-vs-spreadsheet, SR 11-7 governance.

## License

[MIT](LICENSE) © 2026 Willie Man
