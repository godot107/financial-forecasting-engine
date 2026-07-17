# Boardroom Deck — Capital Allocation Under Uncertainty
### Mock CFO / Capital Allocation Committee presentation

This is the slide-by-slide narrative for the executive deck. Each slide lists the
**figure** to show (exported to `../notebooks/figures/`), the **headline numbers**,
and a **speaker script**. Numbers are the canonical run from
[`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) (HMM drivers + QuantLib term
structure + macro SCM); regenerate with `python -m fce --report --hmm`.

> Audience: CFO, VP Corporate Treasury, Capital Allocation Committee.
> Ask: approve the FY CapEx allocation across five initiatives under a $200M budget
> and a $200M CFaR₉₅ liquidity floor.

---

## Slide 1 — Title

**Allocating $200M of CapEx When the Future Is a Distribution, Not a Number**
*A probabilistic capital-allocation engine — auditable accounting, quantified tail risk.*

**Script:** "Every year we commit hundreds of millions across competing projects on
the strength of spreadsheet forecasts that give us a single number per project.
Today I'll show you the same decision made against the *full distribution* of
outcomes — so we can see, and price, the risks a single number hides."

---

## Slide 2 — The problem: the "Excel Wall"

- Static spreadsheets give one NPV per project from single-point assumptions.
- They can't represent **simultaneous shocks** (oil + rates + inflation interacting).
- They hide **tail risk** — the scenarios that breach covenants and drain liquidity.

**Script:** "A spreadsheet says Project Beta is a 22% IRR winner assuming a flat 7%
discount rate and average crack spreads. But rates float, oil is volatile, and
those risks are correlated. The question isn't 'what's the NPV' — it's 'what's the
*distribution* of NPV, and how much of the downside can we survive?'"

---

## Slide 3 — The recommendation

**Figure:** `03_allocation.png`

- **Expected portfolio NPV (P50): $150M**
- **Guaranteed liquidity (CFaR₉₅): $209M** (above the $200M floor)
- Deploys the full $200M: **Beta (LNG) $90M · Epsilon (Chemical) $70M · Alpha (Clean) $40M**; Gamma and Delta not funded this cycle.

**Script:** "This is the return-maximizing allocation that still clears our $200M
liquidity floor in expectation. It concentrates in the high-return energy projects.
That's the *starting* answer — the rest of this deck stress-tests it, because
'optimal in expectation' is not the same as 'safe.'"

---

## Slide 4 — Tails vs. Spreadsheet (the differentiator)

**Figure:** `04_tails_vs_spreadsheet.png`

- The spreadsheet reports one point (~mid-distribution); the engine shows the whole spread.
- The **left 5% tail** — where liquidity collapses and covenants break — is invisible to a single-point model.

**Script:** "Here's the portfolio NPV across 10,000 simulated market states. The dashed
line is what a static model reports. Notice everything to the *left* of it — the red
tail. That mass is real probability the spreadsheet simply cannot see. Our entire
approach exists to put a number on that tail and then constrain it."

---

## Slide 5 — How we model uncertainty: regimes, not random walks

**Figures:** `05_regimes.png`, `07_hmm_vs_gbm.png`

- We fit a Bayesian **regime-switching model** to oil: calm vs. crisis volatility.
- It puts **more mass in the tails** than a naïve model: **1%-worst 3-year drawdown −86% vs. −71%** for a matched Gaussian.

**Script:** "Commodity prices don't drift smoothly — they switch between calm and
crisis regimes, in red here. A single-volatility model understates exactly the
deep-drawdown scenarios that threaten us: our regime model puts the 1-in-100 oil
outcome at minus 86%, versus minus 71% for the naïve model. We would rather plan
against the fatter tail."

---

## Slide 6 — Financing is not a flat number

**Figures:** `08_yield_curve.png`, `10_floating_debt.png`

- Discounting uses a **bootstrapped term structure**, not a flat WACC.
- Project debt is **floating-rate**: interest rises with rates — and hits hardest when crude-exposed revenue is weakest.

**Script:** "Two more things a flat model misses. First, money has a term structure —
we discount each year at its own rate. Second, most of our debt floats: when rates
spike, our interest expense widens *and* our discount rate rises, at the same time
oil-linked revenue is under pressure. These effects compound, and we model them
jointly."

---

## Slide 7 — The real choice: the risk–return frontier

**Figure:** `02_frontier.png`

- Every point is the max expected NPV attainable for a given guaranteed liquidity.
- **Buying safety is priced:** moving CFaR₉₅ from **$209M to $260M costs ~$41M of expected NPV.**

**Script:** "This reframes the decision. It's not 'what's the NPV' — it's 'how much
upside are we willing to trade for how much liquidity insurance?' Moving from our
recommended point to the most defensive point on this curve costs about $41M of
expected value. That's a decision for this committee, and now it's quantified."

---

## Slide 8 — Scenario stress on the recommended plan

**Figure:** `14_scenarios.png`

| Scenario | Expected NPV | CFaR₉₅ | Breach? |
|---|---:|---:|:--:|
| Baseline | $150M | $209M | — |
| SOFR +200bp | $120M (−$29M) | $196M | ⚠️ below floor |
| WTI −30% | $50M (−$100M) | $152M | ⚠️ below floor |
| Rate spike + oil crash | $28M (−$122M) | $142M | ⚠️ below floor |
| Soft landing (SOFR −100bp) | $165M (+$15M) | $215M | — |

**Script:** "Under our recommended plan: a rate spike costs $29M and dips us below the
liquidity floor. An oil crash costs $100M. Both together — the tail the spreadsheet
never shows — costs $122M and takes CFaR to $142M. This is why we don't stop at the
expected-value answer."

---

## Slide 9 — Reverse stress: what actually breaks us?

**Figure:** `16_reverse_stress.png`

- Instead of "what if scenario X," we search the shock space for the **covenant breach frontier**.
- Oil-price crashes dominate rate hikes — the black line is where liquidity fails.

**Script:** "The most useful question is the inverse one: what *combination* of shocks
breaks our covenant? This map answers it. The black line is the breach frontier —
everything to its lower-right is a liquidity failure. And notice it's far steeper in
the oil direction than the rate direction: our exposure is fundamentally to the
commodity. The dangerous shocks are the ones *not* in our history."

---

## Slide 10 — Why our scenarios are *causally* correct

**Figures:** `12_macro_dag.png`, `13_do_vs_condition.png`

- Scenarios are applied as **interventions** (`do`), not by filtering history (conditioning).
- Conditioning would **overstate the inflation→demand effect by ~109%** here, because rates confound it — you'd hedge the wrong risk.

**Script:** "One subtle but important point for model risk. When we say 'set rates
+200bp,' we *intervene* — we set rates and let the true downstream effects follow.
The naïve alternative, filtering history for high-rate periods, drags in everything
correlated with high rates and, here, overstates the demand impact by over 100%.
Getting the counterfactual right is the difference between hedging the real risk and
an imagined one."

---

## Slide 11 — Auditability & model governance (SR 11-7)

- **Deterministic, audited accounting.** Every one of the 10,000 scenarios produces a
  balanced 3-statement articulation (assets = liabilities + equity; cash ties to the
  cash-flow statement). Guarded by automated tests on every run.
- **Uncertainty is quarantined upstream** in the drivers; the accounting math never
  changes for the same inputs — GAAP-compliant and reproducible.
- **Stress & sensitivity testing** (this deck) is itself SR 11-7 evidence that the
  model is conservative.
- **Explainability:** driver attribution via SHAP (planned) explains *which* macro
  variable drove each project's distribution; the tornado attributes NPV movement in
  the accounting output. Different questions, both answered.

**Script:** "Internal audit's first question is always 'can we trust the numbers.' The
answer is structural: the accounting is 100% deterministic and every simulated
balance sheet ties out — we test that automatically. All the uncertainty lives in the
driver models, which are documented, validated against known cases, and explainable.
This deck's stress tests *are* the governance evidence."

---

## Slide 12 — The ask

- Approve the recommended allocation **(Beta $90M · Epsilon $70M · Alpha $40M)**, **or**
- Select a more defensive point on the frontier (trade ~$41M NPV for ~$51M more guaranteed liquidity), **and**
- Set the hard covenant floor that anchors the reverse-stress boundary.

**Script:** "So the decision in front of us is threefold: approve the return-maximizing
plan, or move to a more resilient point on the frontier we've now priced; and confirm
the covenant floor we manage against. Whatever we choose, we're choosing it with the
tail in full view — not hidden inside a single number."

---

## Boardroom Q&A — anticipated questions

**"Can we trust the accounting?"**
Yes — structurally. The 3-statement engine balances by construction and is checked on
every run (assets = liabilities + equity, cash ties to the cash-flow statement,
retained earnings roll forward). Nothing stochastic touches the accounting; all
uncertainty is upstream in the drivers.

**"Why not just use our spreadsheet / why is this better?"**
The spreadsheet gives one number and hides the tail (Slide 4). It also can't represent
correlated shocks or floating-rate/term-structure effects. We reproduce the
spreadsheet's point estimate *and* the distribution around it.

**"These are synthetic projects — is the model real?"**
The five projects are illustrative, but the machinery is production-grade: the driver
model is validated by recovering known parameters from synthetic data; the accounting
is unit-tested; the optimizer and curves use standard libraries (CVXPY, QuantLib). Swap
in real project cash-flow templates and it runs unchanged.

**"How do you pick the discount rate?"**
We don't pick one — we bootstrap a term structure and simulate rates around it, then
discount each cash flow pathwise. A project spread over the risk-free path stands in
for WACC; a full CAPM build is a straightforward extension.

**"What's the model-risk exposure (SR 11-7)?"**
Documented and mitigated: sensitivity + stress testing (this deck) demonstrates
conservatism; scenario counterfactuals are causally correct (interventions, not
conditioning); driver models are validated by parameter recovery; assumptions (the
macro DAG, coefficients) are explicit and reviewable. No causal *discovery* is
claimed — the graph is asserted and auditable.

**"Why is the recommended plan so concentrated in energy projects?"**
Because it maximizes expected NPV subject only to the liquidity floor. The stress
slides show the cost of that concentration; the frontier (Slide 7) lets the committee
choose a more diversified, lower-return, higher-resilience point deliberately.

**"What would change your recommendation?"**
A tighter covenant, a house view that assigns more probability to an oil crash, or a
lower risk appetite — all of which move us left along the frontier toward the battery
(Gamma) hedge, which is negatively correlated with crude.

---

## Appendix — methodology & grounding

- **Drivers (Pillar 1):** Bayesian Gaussian HMM fit with NUTS; forward-algorithm
  marginalization; posterior-predictive simulation. *(Dixon et al., Ch. 7; Jurafsky &
  Martin, forward algorithm.)*
- **Term structure (Pillar 2):** QuantLib bootstrap (deposit + swap helpers); Vasicek
  short rates; floating-rate debt at Act/360. *(Ma Weiming, Ch. 5; Berk & DeMarzo,
  App. 6A.)*
- **Accounting (Pillar 3):** vectorized, balanced 3-statement engine; NPV = −C₀ +
  PV(FCFF). *(Berk & DeMarzo, Eq. 8.6; Hilpisch, Ch. 16–17.)*
- **Allocation (Pillar 4):** CVXPY convex program; CFaR via the Rockafellar–Uryasev
  CVaR linearization. *(Handbook of Energy Trading, p. 196; Rockafellar & Uryasev,
  2000.)*
- **Scenarios:** `do()`-interventions over a hand-specified macro SCM. *(Peters et al.,
  Ch. 6; Molak, p. 89.)*
- **Validation:** purged + embargoed rolling-origin CV. *(López de Prado, AFML Ch. 7.)*

*Reproduce every headline number: `python -m fce --report --hmm`. Full engine:
`python -m fce --allocate --hmm --quantlib`.*
