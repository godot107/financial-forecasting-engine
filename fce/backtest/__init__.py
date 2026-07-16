"""Pillar 4 (validation) — leakage-free backtesting and calibration.

Two *separate* loops, kept distinct on purpose (locked day-1 decision):

* **Statistical loop** — purged + embargoed rolling-origin CV on the *real*
  observable drivers (WTI, rates, inflation), scored by interval coverage and
  pinball loss. This is where López de Prado, AFML Ch. 7 applies.
* **Policy loop** — replay the CVXPY allocation over historical driver
  realizations vs. IRR / equal-weight baselines. Deterministic given drivers;
  no pinball loss here.
"""

from fce.backtest.coverage import coverage, pinball_loss
from fce.backtest.splits import purged_embargoed_splits

__all__ = ["purged_embargoed_splits", "coverage", "pinball_loss"]
