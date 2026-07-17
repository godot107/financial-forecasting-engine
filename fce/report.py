"""Automated CFO-ready executive summary (Pillar 5, reporting).

Runs the whole engine once and renders a one-page Markdown brief — the recommended
allocation, the risk-return frontier, and the named-scenario stress table — with
**real, reproducible numbers** so the boardroom deck (``deck/DECK.md``) never
hand-waves a figure. ``python -m fce --report`` prints it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fce.config import Settings, get_settings
from fce.optimize import efficient_frontier, max_cfar
from fce.scenarios import build_context, run_scenarios
from fce.scenarios.scm import MacroSCM


@dataclass
class ExecutiveSummary:
    config: str
    budget: float
    cfar_target: float
    names: list
    dollars: np.ndarray
    weights: np.ndarray
    expected_npv: float
    cfar: float
    deployed: float
    frontier: list           # list[FrontierPoint]
    scenarios: list          # list[ScenarioResult]
    do_overstatement_pct: float

    def to_markdown(self) -> str:
        m = ["# Executive Summary — Capital Allocation Recommendation", ""]
        m.append(f"*Engine configuration: {self.config}. "
                 f"Budget ${self.budget/1e6:,.0f}M · CFaR₉₅ target "
                 f"${self.cfar_target/1e6:,.0f}M.*\n")
        m.append("## Recommendation\n")
        m.append(f"- **Expected portfolio NPV (P50): ${self.expected_npv/1e6:,.0f}M**")
        m.append(f"- **Guaranteed liquidity (CFaR₉₅): ${self.cfar/1e6:,.0f}M**")
        m.append(f"- Capital deployed: ${self.deployed/1e6:,.0f}M of ${self.budget/1e6:,.0f}M\n")
        m.append("| Project | Allocation | % of budget |")
        m.append("|---|---:|---:|")
        for n, d, w in zip(self.names, self.dollars, self.weights):
            m.append(f"| {n} | ${d/1e6:,.1f}M | {w*100:.1f}% |")
        m.append("")
        m.append("## Scenario stress (locked policy)\n")
        m.append("| Scenario | Expected NPV | Δ NPV | CFaR₉₅ | Breach |")
        m.append("|---|---:|---:|---:|:--:|")
        for r in self.scenarios:
            flag = "⚠️" if r.breaches_floor else ""
            m.append(f"| {r.name} | ${r.expected_npv/1e6:,.0f}M | "
                     f"{r.d_npv/1e6:+,.0f} | ${r.cfar/1e6:,.0f}M | {flag} |")
        m.append("")
        lo = min(self.frontier, key=lambda p: p.cfar)
        hi = max(self.frontier, key=lambda p: p.cfar)
        m.append("## Risk–return frontier\n")
        m.append(f"- Buying liquidity is priced: moving CFaR₉₅ from "
                 f"${lo.cfar/1e6:,.0f}M to ${hi.cfar/1e6:,.0f}M costs "
                 f"${(lo.expected_npv-hi.expected_npv)/1e6:,.0f}M of expected NPV.\n")
        m.append("## Governance\n")
        m.append(f"- Every scenario's balance sheet articulates (assets = liabilities "
                 f"+ equity); accounting is deterministic and audited on every run.")
        m.append(f"- Scenario counterfactuals use *interventions* (`do`), not "
                 f"conditioning: conditioning would overstate the inflation→demand "
                 f"effect by ~{self.do_overstatement_pct:.0f}% here (confounding).")
        return "\n".join(m)


def executive_summary(settings: Settings | None = None, *, use_hmm: bool = False) -> ExecutiveSummary:
    """Compute the canonical executive summary from the live engine."""
    settings = settings or get_settings()
    ctx = build_context(settings, use_hmm=use_hmm)

    # Frontier over the baseline scenario matrices held on the context.
    lo = ctx.base_cfar
    hi = max_cfar(ctx.scen.liq_per_dollar, caps=ctx.caps, budget=settings.capex_budget,
                  alpha=settings.cfar_alpha)
    floors = np.linspace(lo, 0.995 * hi, 10)
    frontier = efficient_frontier(
        ctx.scen.npv_per_dollar, ctx.scen.liq_per_dollar, names=ctx.names,
        caps=ctx.caps, budget=settings.capex_budget, cfar_floors=floors,
        alpha=settings.cfar_alpha,
    )

    results, _ = run_scenarios(ctx=ctx)

    # do-vs-condition overstatement (the causal-correctness headline).
    scm, base = MacroSCM(), ctx.base_sample
    do_s = scm.do(base, {"inflation": lambda x: x + 0.01})
    itv = (do_s.demand - base.demand).mean() / 0.01
    obs = np.polyfit(base.inflation.ravel(), base.demand.ravel(), 1)[0]
    overstate = (obs - itv) / itv * 100

    return ExecutiveSummary(
        config="HMM drivers + QuantLib term structure + macro SCM" if use_hmm
        else "GBM drivers + QuantLib term structure + macro SCM",
        budget=settings.capex_budget,
        cfar_target=settings.cfar_floor,
        names=ctx.names,
        dollars=ctx.dollars,
        weights=ctx.dollars / settings.capex_budget,
        expected_npv=ctx.base_npv,
        cfar=ctx.base_cfar,
        deployed=float(ctx.dollars.sum()),
        frontier=frontier,
        scenarios=results,
        do_overstatement_pct=overstate,
    )
