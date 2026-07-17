# Financial Forecasting Engine (`fce`)

Trello card #80 — https://trello.com/c/9XRsQkBd. A production-grade probabilistic
capital-allocation engine for energy/industrial CapEx under macro uncertainty.
Portfolio piece for the energy-industry / AI Solutioning Consultant pivot.

The identity of this project = **probabilistic simulation + optimization** with a
**100 % deterministic, auditable accounting core**. All uncertainty enters
upstream as simulated driver trajectories; the accounting/discounting math never
changes for the same inputs.

## Run / build

```bash
python -m venv .venv && source .venv/bin/activate   # PROJECT-LOCAL venv (decision #4)
pip install -e '.[dev]'          # core + pytest (fast; no heavy pillars)
pip install -r requirements.txt  # full 4-pillar stack when building pillars 1/2/4
python -m fce                    # runs the MVP slice end-to-end (placeholder driver)
pytest                           # golden accounting-identity + split-leakage tests
```

## Architecture — 4 pillars + validation

| Pillar | Package | Status |
|---|---|---|
| 1. Probabilistic drivers (NumPyro HMM/state-space) | `fce/drivers/` | **built + tests** (GBM still default in pipeline) |
| 2. Term structure + debt (QuantLib) | `fce/term_structure/` | **built + tests** (flat WACC still default) |
| 3. Deterministic 3-statement accounting (NumPy→JAX) | `fce/accounting/` | **built + golden tests** |
| 4. Stochastic allocation under CFaR (CVXPY) | `fce/optimize/` | **built + tests** |
| Validation: purged/embargo CV + policy replay | `fce/backtest/` | splits/metrics built; policy stub |
| What-if / SCM scenarios | `fce/scenarios/` | **built + tests** (do()-interventions) |
| Ingest (EIA + FRED, pinned parquet) | `fce/ingest/` | **built** |
| 5 synthetic projects → scenario sim | `fce/projects.py` | **built** (feeds optimizer) |

Two entrypoints in `fce/pipeline.py` (orchestrator-agnostic, same convention as
`energy-batch-trader`): `run_pipeline()` = MVP accounting slice (Pillar 3);
`run_allocation()` = full CFaR-constrained allocation + efficient frontier
(Pillars 3+4). CLI: `python -m fce` / `python -m fce --allocate`.

### Scenarios / SCM details (built)
- `fce/scenarios/scm.py` — `MacroSCM`: hand-specified DAG (rates→{inflation,demand,
  discount}, inflation→demand, commodity/demand→revenue) + structural equations.
  `simulate()` → `MacroSample`; `do(sample, {node: fn})` sets intervened nodes and
  recomputes descendants **holding exogenous noise fixed** (proper counterfactual).
  Builders: `rate_shock(bp)`, `commodity_shock(pct)`, `combine()`.
- `fce/scenarios/whatif.py` — `build_context()` locks the recommended allocation on
  the baseline world; `run_scenarios()` (named), `tornado()`, `reverse_stress()` all
  score that **locked policy** under `do()`-shocked drivers via `optimize.evaluate()`.
- Integration: `simulate_project_scenarios(demand_paths=…)` feeds the SCM `demand`
  node into revenue; rates→discount/debt already wired (Pillar 2). Confounding
  triangle (rates→inflation, rates→demand) makes do vs. condition genuinely differ.
- **Design note:** the recommended policy sits *at* the CFaR target by construction,
  so `reverse_stress` breaches against a lower **covenant floor** (default 0.85×base
  CFaR) — else the breach frontier degenerates to "everything".
- **SCOPE GUARD holds:** interventional semantics only; NO causal discovery / effect
  estimation (that's the retail-pricing project).

### Pillar 2 details (built)
- `fce/term_structure/curves.py` — **QuantLib** bootstrap: deposit helpers (≤1y) +
  swap helpers (>1y) → `PiecewiseLogLinearDiscount`. `TermStructure` wrapper gives
  `.discount()/.zero_rate()/.forward_rate()/.monthly_discount_factors()`.
- `fce/term_structure/rates.py` — **Vasicek** short-rate sim calibrated to the
  curve (r0=front, θ=10y zero); `discount_factors_from_short_rates()` = pathwise
  DFs; `rate_scenario_paths(settings)` ties curve+sim → `(short_rates, curve)`.
- `fce/term_structure/debt.py` — floating-rate interest = outstanding·(r+spread)·τ
  with **Actual/360** accruals (`monthly_accruals`); bullet/straight amortization.
- `fce/term_structure/history.py` — `load_treasury_curve()` = FRED DGS* or offline
  synthetic par curve.
- Accounting gained `present_value(fcff, discount_factors)` (DF-based, general form
  of `npv`). `simulate_project_scenarios(short_rates=…)` → floating debt + pathwise
  discounting; per-project `credit_spread` added to `Project`.
- **Integration:** `run_allocation(use_quantlib=True)` / `python -m fce --allocate
  --quantlib` (composes with `--hmm`). Default stays flat WACC + fixed coupon.
- QuantLib API notes: `SwapRateHelper` needs an `IborIndex`; `curve.calendar()` can
  be null (compute forwards via explicit `date + Period`). `Actual360`, US
  GovernmentBond calendar.

### Pillar 1 details (built)
- `fce/drivers/hmm.py` — Gaussian **HMM on WTI monthly log-returns**, fit with
  **NUTS** (discrete states marginalized via the **forward algorithm** in
  `lax.scan`; **ordered log-scales** break label-switching → state 0 = calm,
  last = crisis). `fit_hmm` → `HMMPosterior`; `simulate_from_posterior` =
  posterior-predictive forward sim (each path draws a full param set → parameter
  uncertainty in the fan); `viterbi` + `filtered_last_state` for decoding.
- `fce/drivers/history.py` — `load_wti_monthly()` (real EIA→monthly, else offline
  **synthetic 2-regime series**; the synthetic generator = ground truth for the
  recovery test). No API key needed to run.
- **Integration:** `wti_scenario_paths(settings)` fits once + **caches the
  posterior** to `data/hmm_posterior.<vintage>.npz` (gitignored), then simulates
  fast. `simulate_project_scenarios(wti_paths=..., wti0=...)` injects it.
  `run_allocation(use_hmm=True)` / `python -m fce --allocate --hmm` opts in;
  **default stays GBM** so the CLI is snappy. HMM run ≈17s first (fit), ≈7s cached.
- Watch: NUTS in tests (~13s total) — `test_drivers.py` shares ONE fit via a
  module fixture. `$` in Plotly titles triggers LaTeX (mangles them) — avoid.

### Pillar 4 details (built)
- `fce/optimize/allocate.py::allocate()` — continuous LP (locked decision #3),
  maximizes E[portfolio NPV] s.t. budget, per-project absorption caps, and a
  **CFaR floor** via the **Rockafellar–Uryasev CVaR linearization**. Dollar
  quantities are internally normalized to $M — raw ~1e8 magnitudes across 10k
  scenarios break conic-solver conditioning (spurious "infeasible"). Solvers:
  CLARABEL/SCS (ECOS not installed).
- `fce/optimize/frontier.py` — `efficient_frontier()` sweeps CFaR floors;
  `max_cfar()` finds the safe endpoint. **Sweep BETWEEN the max-NPV CFaR (left
  end) and max_cfar (right end)** — sweeping below the max-NPV CFaR is all-slack
  and gives a degenerate flat frontier.
- `fce/projects.py` — 5 synthetic projects with distinct WTI betas (Gamma < 0 =
  genuine hedge) and PERSISTENT per-scenario idio factors (iid-per-month washes
  out over the horizon → correlations ≈ 1). NPV nets upfront capital. Tuned so
  high-return projects (Beta/Epsilon) carry fat left tails and the frontier
  actually slopes.

## Notebooks (tutorial + demo, grounded against textbook-kb)

Standing workflow: each pillar gets an executable tutorial notebook whose claims
are reviewed against the textbook-kb MCP (cite inline + References section).
- `notebooks/01_deterministic_accounting_core.ipynb` — Pillar 3 (FCFF, balances-
  by-construction, golden identities, MC NPV). matplotlib.
- `notebooks/02_capital_allocation_cfar.ipynb` — Pillar 4 (risk-return scatter,
  efficient frontier, recommended allocation, "Tails vs. Spreadsheet"). **Plotly**
  (interactive) + exports `notebooks/figures/*.png` for the NotebookLM/PowerPoint
  deck. Rebuild+execute: `jupyter nbconvert --to notebook --execute --inplace <nb>`.
- `notebooks/03_probabilistic_drivers_hmm.ipynb` — Pillar 1 (Viterbi-decoded
  regimes, posterior-predictive fan, HMM-vs-GBM fatter-tails). Plotly + PNGs.
- `notebooks/04_term_structure_quantlib.ipynb` — Pillar 2 (bootstrapped curve,
  Vasicek short-rate fan, floating-debt sensitivity, discounting impact). Plotly + PNGs.
- `notebooks/05_scenarios_causal_stress.ipynb` — Scenarios (macro DAG, do-vs-
  conditioning, named scenarios, tornado, reverse-stress heatmap). Plotly + PNGs.
- Deck path: Plotly for the dashboard; PNGs → NotebookLM/PowerPoint for the deck.
- Notebook tooling (`nbformat nbconvert ipykernel matplotlib plotly kaleido`) is
  installed in `.venv` but NOT in `pyproject`/`requirements` yet.

## Locked pre-build decisions (from the card's Pre-Build Readiness comment)

1. **MVP = one project / one driver** (WTI→FCFF→NPV→single allocation) end-to-end
   before breadth. The 5 synthetic projects (Alpha–Epsilon) come later.
2. **NumPyro only, drop Stan** initially (rides JAX; keep "Stan-ready" as talk).
3. **Continuous convex allocation** (LP/QP, ECOS/SCS); binary/MIP optional later.
4. **Project-local `.venv`** — the QuantLib+JAX+NumPyro+CVXPY stack is heavy.

## Day-1 invariants (do not regress)

- **Two SEPARATE validation loops.** Statistical loop = purged+embargo rolling
  origin + coverage/pinball on *real* drivers (WTI, rates, inflation), AFML Ch. 7.
  Policy loop = replay allocation vs. IRR/equal-weight baselines; deterministic
  given drivers, **no pinball loss**. Kept as distinct modules in `fce/backtest/`.
- **Golden 3-statement articulation tests** (`assert_balanced`: BS balances, cash
  ties to CF, RE rolls forward) run in CI *and* inside `run_pipeline()` every run.
- **CFaR as a convex constraint via Rockafellar–Uryasev** LP linearization.
- **Cached parquet, pinned vintage** (`Settings.data_vintage`) so 2020/2022 stress
  backtests reproduce offline.

## Build order (accounting BEFORE drivers)

ingest → **accounting + golden tests** (done) → drivers → term structure →
optimize → backtest/policy → exec deck. Build a deterministic skeleton you trust,
*then* inject uncertainty. The MVP pipeline currently uses a placeholder GBM WTI
driver so the accounting core is exercised end-to-end now.

## Implementation notes

- **NumPy today, JAX-ready.** `fce/accounting` uses `numpy` for zero-friction
  testing; the API is a `jax.numpy` drop-in. Swap the import (+`jax.vmap` over
  paths) for the JIT/vectorized 10k-path execution once the heavy venv is built.
- **Reuse:** EIA WTI client is lifted/trimmed from
  `../energy-batch-trader/energy_trader/eia.py`. If a 2nd consumer appears,
  promote `fce/ingest/` to `../shared/`.
- **Causal scope guard (`fce/scenarios/scm.py`):** add ONLY interventional
  scenario semantics — `do()` over a *hand-specified* macro DAG. **Do NOT build**
  causal discovery (PC/GES/NOTEARS) or treatment-effect estimation (ATE/DiD/IV):
  shaky identifiability, synthetic entities have no real treatment, and it
  overlaps the dedicated retail-pricing causal project.

## Grounding (textbook KB, on the card)

HMM/state-space = Dixon *ML in Finance* Ch. 7; MC discounting = Hilpisch *Python
for Finance* Ch. 16–17; backtest leakage fix = López de Prado *AFML* Ch. 7;
CFaR>VaR = *Handbook of Energy Trading* p. 196; what-if/SR 11-7 governance =
Berk & DeMarzo, Mack, Edwards; interventional semantics = Pearl / Peters et al.
*Elements of Causal Inference* §6.2–6.3 / Molak.
