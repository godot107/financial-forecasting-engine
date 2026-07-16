"""Hand-specified macro Structural Causal Model for interventional scenarios.

The edges are *asserted*, not learned (no causal discovery — see the scope guard
in the package docstring). A minimal, defensible DAG over the ~5 macro drivers::

    rates → inflation → demand
    rates → discount_factor
    commodity → revenue

A what-if like "set SOFR +200bp" is a ``do(rates := rates+200bp)`` intervention:
we surgically fix ``rates`` and let its causal *children* (inflation, demand,
discount) respond, while variables that are merely *correlated* with high rates
are left alone. Naively conditioning the driver joint would drag those correlated
variables along and produce the wrong counterfactual (Pearl `do(x)` vs `p(y|x)`;
Peters et al., *Elements of Causal Inference* §6.2–6.3).

STUB — the DAG structure is declared; the intervention propagation is TODO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Asserted macro edges (parent → children). Extend deliberately, never by search.
DEFAULT_EDGES: dict[str, list[str]] = {
    "rates": ["inflation", "discount_factor"],
    "inflation": ["demand"],
    "commodity": ["revenue"],
}


@dataclass
class MacroSCM:
    """A hand-specified macro DAG with a ``do()`` intervention operator."""

    edges: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_EDGES.items()}
    )

    def children(self, node: str) -> list[str]:
        """Direct causal children of ``node``."""
        return list(self.edges.get(node, []))

    def do(self, samples, interventions: dict[str, float]):
        """Apply ``do(interventions)`` to a driver-sample structure.

        Surgically set the intervened drivers and propagate to their causal
        children through the structural equations (leaving non-descendants at
        their observational draws). ``samples`` is the driver simulation to
        intervene on; ``interventions`` maps driver name → clamped value/shift.
        """
        raise NotImplementedError(
            "do()-intervention propagation is a Phase-4 deliverable. The DAG "
            "structure is defined; wire the structural equations next."
        )
