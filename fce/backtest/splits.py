"""Purged + embargoed rolling-origin cross-validation (López de Prado, AFML Ch. 7).

A naive rolling split leaks here: multi-horizon cash-flow labels *overlap* (the
label at origin ``T_k`` draws on the same forward window as ``T_{k+1}``) and the
drivers are serially correlated, so information bleeds across the split boundary.
The fix is two steps:

1. **Purge** — drop training origins whose forward label window ``[t, t+h)``
   overlaps the test window (Fig. 7.2).
2. **Embargo** — additionally drop a fraction of bars immediately *after* the
   test set (Fig. 7.3), since ARMA-like serial correlation leaks even without
   direct label overlap.

This module yields index arrays only — it is model-agnostic and pure, so it is
unit-testable without any of the heavy pillar dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    """One rolling-origin fold: integer index arrays into the time axis."""

    train: np.ndarray
    test: np.ndarray


def purged_embargoed_splits(
    n: int,
    *,
    horizon: int,
    n_splits: int = 5,
    embargo_frac: float = 0.01,
):
    """Yield expanding-window :class:`Split`s with purge + embargo applied.

    ``n`` is the number of time steps; ``horizon`` is the forward label length
    ``h`` (the source of overlap). Each fold trains on an expanding prefix and
    tests on the next contiguous block; training indices whose label window
    ``[i, i+horizon)`` reaches into the test block, plus an embargo of
    ``ceil(embargo_frac * n)`` bars after it, are purged.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    embargo = int(np.ceil(embargo_frac * n))
    # Reserve the first block as the minimum initial training window.
    fold_size = n // (n_splits + 1)
    if fold_size < 1:
        raise ValueError("not enough samples for the requested n_splits")

    for k in range(1, n_splits + 1):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_splits else n
        test = np.arange(test_start, test_end)

        # Candidate training set is everything strictly before the test block.
        train = np.arange(0, test_start)
        # Purge: a training origin i is contaminated if its label window
        # [i, i+horizon) overlaps [test_start, test_end), i.e. i + horizon > test_start.
        train = train[train + horizon <= test_start]
        # Embargo: also drop the last `embargo` bars right before the test block
        # (they are the most serially correlated with it).
        if embargo > 0:
            train = train[train < test_start - embargo]

        yield Split(train=train, test=test)
