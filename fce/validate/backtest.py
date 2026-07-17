"""VaR / CFaR backtesting: are the tail forecasts reliably calibrated?

The number that sells this engine is a tail number (Cash-Flow-at-Risk). A tail
forecast is validated by **exception counting**: over many periods, a 95 % VaR
should be breached ~5 % of the time — no more (model too optimistic) and,
ideally, no fewer (too conservative, wasting capital). Two standard likelihood-
ratio tests formalize this:

- **Kupiec POF** (unconditional coverage): is the *number* of exceptions
  consistent with the stated level? (Kupiec, 1995.)
- **Christoffersen independence**: are exceptions *spread out*, or do they
  cluster — a sign the model misses volatility regimes? (Christoffersen, 1998.)
  Their sum is the conditional-coverage test (both at once).

Because the five projects are synthetic (no realized P&L to backtest), we validate
the *driver's* tail here: walk the HMM forward one step at a time, forecast each
month's VaR from the predictive distribution, and test the exceptions against the
realized WTI returns. The deterministic accounting downstream then inherits that
calibration.

Grounding (textbook-kb): VaR as an exceedance frequency — "the frequency at which
[an] event will occur" and observed-vs-simulated breaches — Davis Edwards,
*Energy Trading & Investing* (2nd ed.), pp. 430–436; tail/worst-case test cases —
*The Handbook of Energy Trading*, §5 (p. 175). Kupiec (1995) and Christoffersen
(1998) supply the formal likelihood-ratio statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, erfc, log, sqrt

import numpy as np

from fce.drivers.hmm import HMMPosterior

_SQRT2 = sqrt(2.0)


def _chi2_sf(x: float, df: int) -> float:
    """Upper-tail probability of a chi-square variate (df = 1 or 2)."""
    if x <= 0.0:
        return 1.0
    if df == 1:
        return erfc(sqrt(x / 2.0))          # P(Z² > x) = erfc(√(x/2))
    if df == 2:
        return float(np.exp(-x / 2.0))      # closed form for df = 2
    raise ValueError("only df in {1, 2} supported")


# --------------------------------------------------------------------------- #
# Exception tests
# --------------------------------------------------------------------------- #
@dataclass
class VarBacktest:
    """Result of an exceedance backtest of a VaR-forecast series."""

    n: int
    exceptions: int
    alpha: float                 # VaR confidence (e.g. 0.95)
    kupiec_stat: float
    kupiec_p: float
    christoffersen_stat: float
    christoffersen_p: float

    @property
    def expected_rate(self) -> float:
        return 1.0 - self.alpha

    @property
    def observed_rate(self) -> float:
        return self.exceptions / self.n if self.n else float("nan")

    @property
    def kupiec_ok(self) -> bool:
        """Not rejected at 5 % — exception count consistent with the level."""
        return self.kupiec_p > 0.05

    def to_markdown(self) -> str:
        return "\n".join([
            f"### VaR backtest (α = {self.alpha:.0%})",
            "",
            f"- Observations: **{self.n}**",
            f"- Exceptions: **{self.exceptions}** "
            f"(observed {self.observed_rate:.1%} vs. expected {self.expected_rate:.1%})",
            f"- Kupiec POF: LR = {self.kupiec_stat:.2f}, p = {self.kupiec_p:.3f} "
            f"→ {'PASS' if self.kupiec_ok else 'REJECT'} (unconditional coverage)",
            f"- Christoffersen CC: LR = {self.christoffersen_stat:.2f}, "
            f"p = {self.christoffersen_p:.3f} "
            f"→ {'PASS' if self.christoffersen_p > 0.05 else 'REJECT'} "
            "(coverage + independence)",
        ])


def kupiec_pof(n: int, x: int, p: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test.

    ``n`` observations, ``x`` exceptions, ``p`` expected exception probability
    (= 1 − confidence). Returns ``(LR_uc, p_value)`` with 1 dof.
    """
    if n <= 0:
        return 0.0, 1.0
    pi = x / n
    ll_null = (n - x) * log(1.0 - p) + (x * log(p) if x else 0.0)
    if x == 0:
        ll_alt = 0.0
    elif x == n:
        ll_alt = n * log(pi)
    else:
        ll_alt = (n - x) * log(1.0 - pi) + x * log(pi)
    lr = -2.0 * (ll_null - ll_alt)
    lr = max(lr, 0.0)
    return lr, _chi2_sf(lr, df=1)


def christoffersen_independence(breaches) -> tuple[float, float]:
    """Christoffersen independence test on a 0/1 breach sequence (1 dof)."""
    b = np.asarray(breaches, int)
    if b.size < 2:
        return 0.0, 1.0
    prev, cur = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    n0, n1 = n00 + n01, n10 + n11
    total = n0 + n1
    if n0 == 0 or n1 == 0 or (n01 + n11) == 0:
        return 0.0, 1.0                      # no clustering signal to test
    pi01, pi11 = n01 / n0, (n11 / n1 if n1 else 0.0)
    pi = (n01 + n11) / total

    def _t(k, q):
        return k * log(q) if (k and q > 0.0) else 0.0

    ll_null = _t(n00 + n10, 1.0 - pi) + _t(n01 + n11, pi)
    ll_alt = _t(n00, 1.0 - pi01) + _t(n01, pi01) + _t(n10, 1.0 - pi11) + _t(n11, pi11)
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    return lr, _chi2_sf(lr, df=1)


def var_backtest(returns, var_forecast, *, alpha: float = 0.95) -> VarBacktest:
    """Backtest a one-step VaR forecast against realized ``returns``.

    ``var_forecast`` is a series of positive loss thresholds aligned with
    ``returns``; a breach is a realized loss exceeding the VaR
    (``returns < -var_forecast``). Runs Kupiec (unconditional) and Christoffersen
    (conditional) coverage tests.
    """
    returns = np.asarray(returns, float)
    var_forecast = np.asarray(var_forecast, float)
    breaches = (returns < -var_forecast).astype(int)
    n, x = int(breaches.size), int(breaches.sum())
    p = 1.0 - alpha

    lr_uc, p_uc = kupiec_pof(n, x, p)
    lr_ind, _ = christoffersen_independence(breaches)
    lr_cc = lr_uc + lr_ind
    return VarBacktest(
        n=n, exceptions=x, alpha=alpha,
        kupiec_stat=lr_uc, kupiec_p=p_uc,
        christoffersen_stat=lr_cc, christoffersen_p=_chi2_sf(lr_cc, df=2),
    )


# --------------------------------------------------------------------------- #
# One-step-ahead VaR from the HMM predictive distribution
# --------------------------------------------------------------------------- #
def _normal_cdf_vec(x: float, locs: np.ndarray, scales: np.ndarray) -> np.ndarray:
    z = (x - locs) / (scales * _SQRT2)
    return 0.5 * (1.0 + np.vectorize(erf)(z))


def _mixture_var(weights, locs, scales, alpha: float) -> float:
    """Loss VaR at level ``alpha`` for a Gaussian mixture return distribution."""
    tail = 1.0 - alpha
    lo = float(locs.min() - 12.0 * scales.max())
    hi = float(locs.max() + 12.0 * scales.max())
    for _ in range(64):                      # bisection on the mixture CDF
        mid = 0.5 * (lo + hi)
        cdf = float(np.sum(weights * _normal_cdf_vec(mid, locs, scales)))
        if cdf < tail:
            lo = mid
        else:
            hi = mid
    return -0.5 * (lo + hi)                   # VaR is a positive loss magnitude


def hmm_one_step_var(returns, post: HMMPosterior, *, alpha: float = 0.95) -> np.ndarray:
    """Walk-forward one-step-ahead VaR from the HMM predictive (posterior-mean).

    At each month the predictive return is a Gaussian mixture over the *predicted*
    regime distribution (filtered belief pushed one step through the transition
    matrix); its lower ``1−alpha`` quantile is the forecast VaR. Returns a VaR
    series aligned with ``returns``, suitable for :func:`var_backtest`.
    """
    returns = np.asarray(returns, float)
    pi, A, locs, scales = post.mean_params()
    belief = pi.copy()                        # predictive state dist before obs 0
    var = np.empty(len(returns))
    inv2 = 1.0 / (2.0 * scales**2)
    norm = 1.0 / (scales * np.sqrt(2.0 * np.pi))
    for t, y in enumerate(returns):
        var[t] = _mixture_var(belief, locs, scales, alpha)
        like = norm * np.exp(-((y - locs) ** 2) * inv2)   # p(y_t | state)
        filt = belief * like
        filt = filt / filt.sum()
        belief = filt @ A                     # predict next month's regime
    return var
