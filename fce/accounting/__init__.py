"""Pillar 3 — deterministic, vectorized 3-statement accounting engine.

This is the auditable core of the project. It takes driver vectors (revenue,
cost inflation, capex) plus a debt/discount schedule and produces fully balanced
Income Statement / Cash Flow / Balance Sheet articulations, then discounts FCFF
to NPV — **deterministically**, so the same inputs always yield the same numbers.

All uncertainty lives upstream in the driver simulation; nothing stochastic
happens here. The golden identity tests (``tests/test_accounting_identities.py``)
guard that promise from the first commit.

.. note::
   Arrays are plain ``numpy`` today for zero-friction testing. The API is a
   ``jax.numpy`` drop-in — swap ``import numpy as np`` for ``import jax.numpy as
   jnp`` (and ``jax.vmap`` over paths) to get the vectorized/JIT execution across
   10,000+ paths described in the card. See ``CLAUDE.md``.
"""

from fce.accounting.identities import assert_balanced
from fce.accounting.statements import (
    Statements,
    build_statements,
    npv,
    present_value,
)

__all__ = ["Statements", "build_statements", "npv", "present_value", "assert_balanced"]
