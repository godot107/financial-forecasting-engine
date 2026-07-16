"""What-if drivers: named scenarios, tornado sensitivity, reverse stress test.

STUB. Interfaces below; each is a thin call over the accounting + MC engine and
the :class:`~fce.scenarios.scm.MacroSCM` do-operator.
"""

from __future__ import annotations


def scenario(name: str, interventions: dict[str, float], engine):
    """Run a named deterministic what-if (e.g. 'SOFR+200_WTI-30').

    ``interventions`` are applied as ``do()`` operations via the macro SCM, then
    pushed through the deterministic accounting layer to yield stressed NPV/CFaR.
    """
    raise NotImplementedError("Named scenario runner not implemented yet.")


def tornado(base_engine, drivers: list[str], shock: float = 0.10):
    """One-at-a-time ±``shock`` on each driver → ranked NPV/CFaR swings.

    Returns the sorted (driver, low, high) rows for a tornado diagram. Note this
    attributes *accounting-output* movement, complementing the SHAP explanation
    of the *driver model* (they answer different questions).
    """
    raise NotImplementedError("Tornado sensitivity not implemented yet.")


def reverse_stress(engine, *, cfar_floor: float, covenant):
    """Solve for the driver combination that breaches the CFaR floor / covenant.

    An optimization over driver space (reuse CVXPY) answering "what breaks us?" —
    the dangerous shocks are the ones *not* in the historical sample.
    """
    raise NotImplementedError("Reverse stress test not implemented yet.")
