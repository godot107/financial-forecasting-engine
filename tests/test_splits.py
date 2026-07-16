"""Tests for the purge + embargo rolling-origin CV (AFML Ch. 7).

The point of purge+embargo is that training origins near the test block are
*removed*, so we assert the leakage gap actually exists.
"""

from __future__ import annotations

import numpy as np

from fce.backtest import purged_embargoed_splits


def test_train_is_strictly_before_test():
    for split in purged_embargoed_splits(200, horizon=6, n_splits=4):
        if split.train.size:
            assert split.train.max() < split.test.min()


def test_purge_removes_overlapping_label_windows():
    horizon = 6
    for split in purged_embargoed_splits(200, horizon=horizon, n_splits=4, embargo_frac=0.0):
        test_start = split.test.min()
        # No training origin's label window [i, i+horizon) may reach the test set.
        assert np.all(split.train + horizon <= test_start)


def test_embargo_widens_the_gap():
    n, horizon = 300, 4
    no_embargo = list(purged_embargoed_splits(n, horizon=horizon, n_splits=5, embargo_frac=0.0))
    with_embargo = list(purged_embargoed_splits(n, horizon=horizon, n_splits=5, embargo_frac=0.05))
    # For a matched fold, the embargoed training set ends earlier (or equal).
    for a, b in zip(no_embargo, with_embargo):
        if a.train.size and b.train.size:
            assert b.train.max() <= a.train.max()


def test_folds_cover_expanding_windows():
    splits = list(purged_embargoed_splits(120, horizon=3, n_splits=5))
    assert len(splits) == 5
    # Test blocks march forward.
    starts = [s.test.min() for s in splits]
    assert starts == sorted(starts)
