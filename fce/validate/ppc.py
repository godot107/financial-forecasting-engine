"""Posterior predictive checks (PPC) for the Gaussian-HMM driver (Pillar 1).

Fitting the model and recovering its parameters tells us the *sampler* worked; it
does not tell us the *model* is adequate for the data. A PPC closes that gap:
because the HMM is generative, we simulate many replicated return series from the
posterior and ask whether features of the *observed* series look like a typical
draw. If an observed test statistic sits far in the tail of its replicated
distribution, the model fails to reproduce that feature.

We check the features that matter for a risk model — dispersion, fat tails,
volatility clustering, and the worst realized move:

- ``vol``       : standard deviation of returns
- ``kurtosis``  : excess kurtosis (fat-tailedness)
- ``acf1_sq``   : lag-1 autocorrelation of squared returns (volatility clustering)
- ``min``       : worst single-period return (left-tail severity)

Each yields a **posterior predictive p-value** = P(T_rep ≥ T_obs). Values near
0.5 indicate the model reproduces the feature; values near 0 or 1 flag misfit.

Grounding (textbook-kb): model checking as generative simulation and its two
goals (did the fit work; is the model adequate) — McElreath, *Statistical
Rethinking* (2nd ed.), Ch. 3 (p. 82); posterior predictive checks to assess and
compare fit — Martin, Kumar & Lao, *Bayesian Modeling and Computation in Python*,
Ch. 2 (p. 62) & Ch. 9 (p. 297); Jansen, *ML for Algorithmic Trading*, Ch. 9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fce.drivers.hmm import HMMPosterior, _categorical_rows


# --------------------------------------------------------------------------- #
# Test statistics (each works on a 1-D series or row-wise on a 2-D ensemble)
# --------------------------------------------------------------------------- #
def _vol(x, axis=-1):
    return np.std(x, axis=axis)


def _kurtosis(x, axis=-1):
    mu = np.mean(x, axis=axis, keepdims=True)
    d = x - mu
    m2 = np.mean(d**2, axis=axis)
    m4 = np.mean(d**4, axis=axis)
    return m4 / np.maximum(m2**2, 1e-30) - 3.0


def _acf1_sq(x, axis=-1):
    x = np.moveaxis(np.asarray(x, float), axis, -1)
    s = x**2
    s = s - s.mean(axis=-1, keepdims=True)
    num = np.sum(s[..., :-1] * s[..., 1:], axis=-1)
    den = np.sum(s * s, axis=-1)
    return num / np.maximum(den, 1e-30)


def _min(x, axis=-1):
    return np.min(x, axis=axis)


_STATS = {
    "vol": _vol,
    "kurtosis": _kurtosis,
    "acf1_sq": _acf1_sq,
    "min": _min,
}


@dataclass
class PPCResult:
    """Observed statistics vs. their posterior-predictive distributions."""

    observed: dict[str, float]
    p_values: dict[str, float]
    rep_mean: dict[str, float]
    n_rep: int

    def worst_pvalue(self) -> float:
        """The most extreme (closest to 0 or 1) p-value across statistics."""
        return min(min(p, 1.0 - p) for p in self.p_values.values())

    def to_markdown(self) -> str:
        lines = [
            "### HMM posterior predictive checks",
            "",
            f"Replicated series: {self.n_rep}. A p-value near 0.5 means the model "
            "reproduces the feature; near 0 or 1 flags misfit.",
            "",
            "| Statistic | Observed | Replicated mean | PPC p-value |",
            "|---|---:|---:|---:|",
        ]
        labels = {
            "vol": "Volatility",
            "kurtosis": "Excess kurtosis",
            "acf1_sq": "Vol clustering (ACF₁ of r²)",
            "min": "Worst monthly return",
        }
        for key in _STATS:
            if key not in self.observed:
                continue
            lines.append(
                f"| {labels.get(key, key)} | {self.observed[key]:.4f} | "
                f"{self.rep_mean[key]:.4f} | {self.p_values[key]:.3f} |"
            )
        return "\n".join(lines)


def _simulate_return_ensemble(post: HMMPosterior, n_rep: int, length: int, rng) -> np.ndarray:
    """Draw ``n_rep`` replicated return series of ``length`` from the posterior.

    Each replicate draws its own parameter set from the posterior (propagating
    parameter uncertainty), then runs the HMM generative process forward.
    """
    d = post.locs.shape[0]
    idx = rng.integers(0, d, size=n_rep)
    pinit = post.probs_init[idx]     # (R, K)
    trans = post.probs_trans[idx]    # (R, K, K)
    locs = post.locs[idx]            # (R, K)
    scales = post.scales[idx]        # (R, K)

    rows = np.arange(n_rep)
    state = _categorical_rows(rng, pinit)
    out = np.empty((n_rep, length))
    for t in range(length):
        out[:, t] = rng.normal(locs[rows, state], scales[rows, state])
        state = _categorical_rows(rng, trans[rows, state])
    return out


def posterior_predictive_check(
    returns,
    post: HMMPosterior,
    *,
    n_rep: int = 300,
    seed: int = 0,
    stats=None,
) -> PPCResult:
    """Run posterior predictive checks of ``post`` against observed ``returns``.

    Simulates ``n_rep`` replicated series (same length as ``returns``) from the
    posterior and compares each test statistic's observed value to its replicated
    distribution. Returns a :class:`PPCResult`.
    """
    returns = np.asarray(returns, float)
    keys = list(stats) if stats is not None else list(_STATS)
    rng = np.random.default_rng(seed)

    rep = _simulate_return_ensemble(post, n_rep, len(returns), rng)  # (R, T)

    observed, p_values, rep_mean = {}, {}, {}
    for key in keys:
        fn = _STATS[key]
        t_obs = float(fn(returns))
        t_rep = np.asarray(fn(rep, axis=1), float)
        observed[key] = t_obs
        rep_mean[key] = float(np.mean(t_rep))
        # Posterior predictive p-value P(T_rep >= T_obs), clipped off the hard 0/1
        # edges so a finite replicate count never reports impossible certainty.
        p = float(np.mean(t_rep >= t_obs))
        p_values[key] = min(max(p, 0.5 / n_rep), 1.0 - 0.5 / n_rep)
    return PPCResult(observed=observed, p_values=p_values, rep_mean=rep_mean, n_rep=n_rep)
