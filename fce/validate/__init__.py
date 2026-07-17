"""Model validation (the "are the *metrics* trustworthy?" layer).

Unit tests prove the code is correct; this package proves the *numbers* are
calibrated — the distinction an SR 11-7 model-risk review turns on. Three
grounded validators, each mapping a metric the engine produces to a textbook
validation technique:

- :mod:`fce.validate.curves`   — does the bootstrapped curve reprice its inputs?
- :mod:`fce.validate.ppc`      — does the HMM reproduce the data's tail/clustering?
- :mod:`fce.validate.backtest` — are the VaR/CFaR tail forecasts calibrated?

Quantile calibration (interval coverage, pinball loss) lives with the driver's
statistical loop in :mod:`fce.backtest.coverage`.
"""

from __future__ import annotations

from fce.validate.backtest import (
    VarBacktest,
    christoffersen_independence,
    hmm_one_step_var,
    kupiec_pof,
    var_backtest,
)
from fce.validate.curves import RepricingResult, reprice_curve
from fce.validate.ppc import PPCResult, posterior_predictive_check

__all__ = [
    "reprice_curve",
    "RepricingResult",
    "posterior_predictive_check",
    "PPCResult",
    "var_backtest",
    "VarBacktest",
    "kupiec_pof",
    "christoffersen_independence",
    "hmm_one_step_var",
]
