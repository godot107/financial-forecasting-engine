# When the Forecast Is a Distribution, Not a Number

*Building a probabilistic capital-allocation engine that keeps the accounting
auditable and puts a price on tail risk.*

---

Every year, companies commit hundreds of millions of dollars across competing
projects on the strength of spreadsheet forecasts. Each project gets one row, one
set of assumptions, and one number at the end: an NPV, an IRR, a payback period.
The committee compares the numbers, funds the winners, and moves on.

The problem is that the single number is a fiction of precision. It assumes a flat
discount rate that won't stay flat, an oil price that won't spike, and cost
inflation that won't arrive — and, crucially, it assumes those things won't all go
wrong *at the same time*. The one scenario a spreadsheet can never show you is the
left tail: the correlated shock that drains liquidity and breaches a debt covenant.
That's exactly the scenario a CFO needs to see.

So I built an engine that models the whole distribution instead of one point on it —
and then makes the allocation decision *against* that distribution. This post walks
through what it does, the one architectural decision that makes it trustworthy, and
what it produces for a decision-maker.

You can find the full, tested, open-source project here:
**[github.com/godot107/financial-forecasting-engine](https://github.com/godot107/financial-forecasting-engine)**

---

## The architectural bet: quarantine the uncertainty

The instinct when you add "probabilistic" to a finance model is to make everything
stochastic. That's a mistake. In corporate finance, some things *must* be exact —
the accounting identities, the discounting arithmetic. If your balance sheet doesn't
balance, no amount of Bayesian sophistication upstream will save you, and internal
audit will (rightly) throw the whole thing out.

So the engine draws a hard line:

- **Accounting and discounting are 100% deterministic and auditable.** A three-
  statement model that balances *by construction* — every simulated scenario
  produces a set of statements where Assets ≡ Liabilities + Equity, exactly, and an
  automated test (`assert_balanced`) checks it on every run.
- **Uncertainty lives only upstream, in the market drivers** — oil prices, interest
  rates, demand. That's where it belongs.

The payoff is the best of both worlds: the reproducibility and audit trail of a
GAAP-style spreadsheet, with the honest tail-risk quantification of a Monte Carlo
engine. For the same inputs, the accounting math never drifts. All the randomness is
in *what the inputs might be*.

This is also the answer to the first question any risk committee asks — *"Can we
trust the numbers?"* — and the answer is structural, not a promise.

---

## Four pillars

The engine is built in four layers, each using the right tool for the job.

### 1. Drivers that model regimes, not random walks
Commodity prices don't drift smoothly — they switch between calm and crisis
volatility. I model oil with a **Bayesian regime-switching hidden Markov model**
(NumPyro / JAX), fit with NUTS, with the discrete states marginalized out via the
forward algorithm. The result matters: it puts materially more mass in the tails
than the "single-volatility" model most spreadsheets implicitly assume. In numbers,
the model's 1-in-100 three-year oil drawdown is **−86%**, versus **−71%** for a
matched Gaussian random walk. When you're stress-testing, you want to plan against
the fatter, more honest tail.

### 2. Financing as a term structure, not a flat rate
A flat WACC hides two things. First, money has a term structure — so I bootstrap a
yield curve (QuantLib) and discount each year's cash flow at its own rate. Second,
most project debt *floats*: when rates spike, interest expense widens **and** the
discount rate rises, at the same moment oil-linked revenue is under pressure. These
effects compound, so the engine models them jointly (Vasicek short-rate simulation
feeding Actual/360 floating-rate debt service).

### 3. Accounting that proves it balances
The deterministic core: a vectorized three-statement engine (Revenue → FCFF → NPV)
that articulates across 10,000+ scenarios at once. Golden identity tests run both in
CI and inside the pipeline. This is the auditable foundation everything else stands
on.

### 4. Allocation that respects a liquidity floor
Finally, the decision. Instead of eyeballing NPVs, capital is allocated by a
**convex optimizer** (CVXPY) that maximizes expected portfolio NPV subject to the
budget, per-project caps, and — the key constraint — a **Cash-Flow-at-Risk floor**.
CFaR is encoded via the Rockafellar–Uryasev CVaR linearization, so "keep at least
$200M of liquidity in the 5%-worst case" becomes a hard, linear constraint the
solver has to satisfy. The recommendation and the risk–return frontier are
*computed*, not asserted.

---

## The differentiator: scenarios that are causally correct

Here's the subtle part, and the piece I'm proudest of.

When a committee asks "what if rates go up 200 basis points?", the naïve way to
answer is to filter history for high-rate periods and look at what happened to
demand. That's **conditioning**, and it's wrong — because rates also drive inflation,
which also drives demand, so filtering on high rates drags in everything correlated
with them. You end up hedging an imagined risk.

The correct way is to treat the scenario as an **intervention** — Pearl's `do()`
operator — over a hand-specified macro DAG: you *set* rates, hold the exogenous
noise fixed, and let only the true downstream effects propagate. In this model, the
difference is not academic: conditioning **overstates the inflation→demand effect by
about 109%** compared to the true causal effect. Getting the counterfactual right is
the difference between hedging the real exposure and hedging a phantom.

(To keep the project honest and focused, I deliberately scoped this to
*interventional* semantics only — no causal discovery, no treatment-effect
estimation. The DAG is asserted and auditable, not learned.)

On top of this sit the tools a treasury team actually uses: named scenarios, a
tornado sensitivity chart, and a **reverse stress test** that searches the shock
space for the covenant-breach frontier — answering the inverse, more useful
question: *what combination of shocks actually breaks us?* (Here, oil crashes
dominate rate hikes — the exposure is fundamentally to the commodity.)

---

## What it produces

Run it end-to-end and you get a decision, not a data dump. The canonical run:

| Metric | Value |
|---|---|
| Expected portfolio NPV (P50) | **$150M** |
| Guaranteed liquidity (CFaR₉₅) | **$209M** (above the $200M floor) |
| Recommended allocation | Beta (LNG) $90M · Epsilon (Chemical) $70M · Alpha (Clean) $40M |

And, more importantly, the **cost of safety made explicit**. The efficient frontier
shows that moving the guaranteed-liquidity floor from $209M up to $260M costs about
**$41M of expected NPV**. That reframes the boardroom conversation entirely. It's no
longer "what's the NPV" — it's "how much upside are we willing to trade for how much
liquidity insurance?" — and now that trade is priced.

Stress-tested against the recommended plan:

| Scenario | Expected NPV | vs. base |
|---|---:|---:|
| Baseline | $150M | — |
| SOFR +200bp | $120M | −$29M |
| WTI −30% | $50M | −$100M |
| Rate spike + oil crash | $28M | −$122M |
| Soft landing (SOFR −100bp) | $165M | +$15M |

That "rate spike + oil crash" row — a $122M swing that pushes liquidity below the
floor — is the tail the spreadsheet never shows. Making it visible, and pricing it,
is the entire point.

The whole thing culminates in a **mock CFO decision deck** framed around SR 11-7
model governance — because a model that a real committee can act on has to be
auditable, validated, and explainable, not just accurate.

---

## Why I built this

I'm making a deliberate pivot toward FP&A leadership and AI solutioning, and I wanted
a portfolio piece that lives at the intersection — one that speaks fluently to both
the finance side (three-statement articulation, NPV/CFaR, covenant risk, model
governance) and the technical side (Bayesian inference, convex optimization, causal
counterfactuals), without hand-waving on either.

The five projects here are synthetic, but the machinery is production-grade: the
driver model is validated by recovering known parameters from synthetic data, the
accounting is unit-tested, and the optimizer and curves use standard industry
libraries. Swap in real project cash-flow templates and it runs unchanged.

It's also, deliberately, **offline-first and reproducible**: no API keys required to
run — the drivers fall back to a reproducible synthetic series — and 38 automated
tests guard the invariants. Every headline number in this post regenerates with a
single command:

```bash
python -m fce --report --hmm
```

The code, the tutorial notebooks (one per pillar, grounded against canonical finance
and ML references), and the boardroom deck are all on GitHub:
**[github.com/godot107/financial-forecasting-engine](https://github.com/godot107/financial-forecasting-engine)**.

The takeaway I keep coming back to: the goal of a forecast isn't a more confident
number. It's an honest distribution — and the discipline to make the decision with
the tail in full view.
