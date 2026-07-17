"""Hand-specified macro Structural Causal Model for interventional scenarios.

The edges are **asserted, not learned** (no causal discovery — see the scope guard
in the package docstring). A small, defensible DAG over the macro drivers::

    rates ──► inflation ──► demand ──► revenue
      │           │          ▲
      │           └──────────┘
      ├──► demand            (rates confounds the inflation→demand relationship)
      └──► discount_factor
    commodity ──► revenue

``rates`` and ``commodity`` are exogenous roots (supplied by Pillars 2 and 1);
``inflation`` and ``demand`` are computed from structural equations with fixed
exogenous noise.

A what-if like *"SOFR +200bp"* is ``do(rates := rates + 0.02)``: we surgically set
``rates`` and let its causal descendants (inflation, demand, discount) respond via
the structural equations, **holding the exogenous noise fixed** — the proper
counterfactual. Variables that are merely correlated with high rates (e.g. the
commodity price) are left untouched. Naive *conditioning* on high rates would drag
correlated variables along and overstate the effect — the classic confounding
error the ``rates → {inflation, demand}`` triangle demonstrates.

Grounding: Pearl's ``do(x)`` vs ``p(y|x)`` — Peters, Janzing & Schölkopf,
*Elements of Causal Inference*, Ch. 6 (do-calculus, p. 137); interventions vs
counterfactuals — Molak, *Causal Inference and Discovery in Python* (p. 89).

**SCOPE GUARD:** interventional semantics only. No causal discovery (PC/GES/NOTEARS)
and no treatment-effect estimation (ATE/DiD/IV) — those belong to the dedicated
retail-pricing project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Asserted macro edges (parent → children). Extend deliberately, never by search.
DEFAULT_EDGES: dict[str, list[str]] = {
    "rates": ["inflation", "demand", "discount_factor"],
    "inflation": ["demand"],
    "commodity": ["revenue"],
    "demand": ["revenue"],
}


@dataclass
class MacroSample:
    """A realized draw of the macro SCM — all arrays ``(S, T)``.

    ``rates`` and ``commodity`` are exogenous; ``inflation`` and ``demand`` are
    structural. The exogenous noise (``eps_*``) is retained so interventions can
    recompute descendants against the *same* world (a proper counterfactual).
    """

    rates: np.ndarray
    commodity: np.ndarray
    inflation: np.ndarray
    demand: np.ndarray
    eps_inflation: np.ndarray = field(repr=False)
    eps_demand: np.ndarray = field(repr=False)


@dataclass
class MacroSCM:
    """A hand-specified macro DAG with structural equations and a ``do`` operator.

    Coefficients are *illustrative and asserted* (the scope is "assume the DAG").
    Signs encode a coherent stress story: a rate spike coincides with higher
    inflation and, both directly and through inflation, weaker demand.
    """

    edges: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_EDGES.items()}
    )
    rate_base: float = 0.045     # baseline short rate
    infl_base: float = 0.025     # baseline inflation
    b_ri: float = 0.50           # rates → inflation (co-move in stress)
    b_di: float = -2.0           # inflation → demand (erodes real demand)
    b_dr: float = -1.5           # rates → demand (direct financial-conditions channel)
    infl_noise: float = 0.003
    dem_noise: float = 0.020

    # --- graph helpers ----------------------------------------------------
    def children(self, node: str) -> list[str]:
        """Direct causal children of ``node``."""
        return list(self.edges.get(node, []))

    def descendants(self, node: str) -> set[str]:
        """All downstream nodes reachable from ``node``."""
        out, stack = set(), list(self.children(node))
        while stack:
            n = stack.pop()
            if n not in out:
                out.add(n)
                stack.extend(self.children(n))
        return out

    # --- structural equations --------------------------------------------
    def _inflation(self, rates, eps):
        return self.infl_base + self.b_ri * (rates - self.rate_base) + eps

    def _demand(self, rates, inflation, eps):
        return (
            1.0
            + self.b_di * (inflation - self.infl_base)
            + self.b_dr * (rates - self.rate_base)
            + eps
        )

    # --- sampling ---------------------------------------------------------
    def simulate(self, rates, commodity, *, seed: int = 0) -> MacroSample:
        """Compute a baseline :class:`MacroSample` from exogenous rates & commodity."""
        rates = np.asarray(rates, float)
        commodity = np.asarray(commodity, float)
        rng = np.random.default_rng(seed)
        eps_i = rng.normal(0.0, self.infl_noise, size=rates.shape)
        eps_d = rng.normal(0.0, self.dem_noise, size=rates.shape)
        inflation = self._inflation(rates, eps_i)
        demand = self._demand(rates, inflation, eps_d)
        return MacroSample(rates, commodity, inflation, demand, eps_i, eps_d)

    # --- the do-operator --------------------------------------------------
    def do(self, sample: MacroSample, interventions: dict) -> MacroSample:
        """Return the counterfactual sample under ``do(interventions)``.

        ``interventions`` maps a node name to a callable ``fn(old_array)->new_array``
        (e.g. ``{"rates": lambda r: r + 0.02}``). Intervened nodes are set directly;
        their descendants are recomputed from the structural equations with the
        **same exogenous noise**; non-descendants are left untouched.
        """
        rates = interventions["rates"](sample.rates) if "rates" in interventions else sample.rates
        commodity = (
            interventions["commodity"](sample.commodity)
            if "commodity" in interventions else sample.commodity
        )

        if "inflation" in interventions:
            inflation = interventions["inflation"](sample.inflation)
        else:
            inflation = self._inflation(rates, sample.eps_inflation)

        if "demand" in interventions:
            demand = interventions["demand"](sample.demand)
        else:
            demand = self._demand(rates, inflation, sample.eps_demand)

        return MacroSample(
            rates, commodity, inflation, demand,
            sample.eps_inflation, sample.eps_demand,
        )


# --- intervention builders (ergonomic named-scenario constructors) --------
def rate_shock(bp: float) -> dict:
    """``do(rates += bp basis points)`` — e.g. ``rate_shock(200)`` = SOFR +200bp."""
    return {"rates": lambda r: r + bp / 1e4}


def commodity_shock(pct: float) -> dict:
    """``do(commodity *= 1 + pct/100)`` — e.g. ``commodity_shock(-30)`` = WTI −30%."""
    return {"commodity": lambda c: c * (1.0 + pct / 100.0)}


def combine(*interventions: dict) -> dict:
    """Merge several intervention dicts into one simultaneous ``do``."""
    out: dict = {}
    for iv in interventions:
        out.update(iv)
    return out
