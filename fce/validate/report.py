"""Compose the three validators into one model-validation report (SR 11-7).

This is the evidence that backs the deck's governance slide: it repriced the
curve, checked the driver reproduces the data, and backtested the tail forecast —
on the actual (real-or-synthetic) data the engine runs on. Regenerate with
``python -m fce --validate``.
"""

from __future__ import annotations

import numpy as np

from fce.config import Settings
from fce.term_structure.history import load_treasury_curve
from fce.validate.backtest import hmm_one_step_var, var_backtest
from fce.validate.curves import reprice_curve
from fce.validate.ppc import posterior_predictive_check


def validation_report(settings: Settings | None = None) -> str:
    """Run all three validators end-to-end and render a Markdown report."""
    settings = settings or Settings()

    # 1. Curve repricing — deterministic round-trip on the loaded par curve.
    tenors, par = load_treasury_curve(settings)
    curve_md = reprice_curve(tenors, par).to_markdown()

    # 2 & 3. Fit the HMM to WTI history, then PPC + one-step VaR backtest.
    from fce.drivers.history import load_wti_monthly
    from fce.drivers.hmm import fit_hmm

    prices = np.asarray(load_wti_monthly(settings), dtype=float)
    returns = np.diff(np.log(prices))
    post = fit_hmm(returns, n_states=settings.hmm_states, seed=settings.seed)

    ppc_md = posterior_predictive_check(returns, post, n_rep=400, seed=settings.seed).to_markdown()
    var = hmm_one_step_var(returns, post, alpha=settings.cfar_alpha)
    var_md = var_backtest(returns, var, alpha=settings.cfar_alpha).to_markdown()

    return "\n\n".join([
        "# Model-Validation Report",
        "*Are the metrics trustworthy? — repricing, adequacy, and tail calibration.*",
        curve_md,
        ppc_md,
        var_md,
        "---",
        "_Unit tests prove the code is correct; this report proves the numbers are "
        "calibrated. Regenerate: `python -m fce --validate`._",
    ])
