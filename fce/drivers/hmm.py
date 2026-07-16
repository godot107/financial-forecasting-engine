"""NumPyro Gaussian-HMM driver model (Pillar 1) — regime-switching WTI.

Replaces the placeholder GBM. A ``K``-state Gaussian hidden Markov model on WTI
monthly log-returns: each latent regime has its own return mean and volatility,
and the chain transitions between regimes (calm ↔ crisis). We:

1. **Fit** the model with NUTS, marginalizing the discrete states via the
   **forward algorithm** so the sampler only touches continuous parameters.
2. **Simulate** forward trajectories by posterior-predictive sampling — each path
   draws a full parameter set from the posterior, so the fan of outcomes carries
   *parameter* uncertainty, not just process noise → honest P10/P50/P90.

Grounding (textbook-kb):
- HMM joint factorization ``p(s,y)=p(s₁)p(y₁|s₁)∏ₜ p(sₜ|sₜ₋₁)p(yₜ|sₜ)`` —
  Dixon, Halperin & Bilokon, *Machine Learning in Finance*, Ch. 7 (pp. 243–244).
- Forward recursion ``αₜ(j)=Σᵢ αₜ₋₁(i) aᵢⱼ bⱼ(oₜ)`` — Jurafsky & Martin, *Speech
  and Language Processing*, Ch. 6 (p. 203); Bishop, *PRML*, §13.2.
- Viterbi "most likely explanation" — Nielsen, *Practical Time Series Analysis*.

**Locked decision #2: NumPyro only** (rides the JAX the accounting layer already
needs; Stan intentionally dropped).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

_LOG_2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- #
# Posterior container + driver-simulation output
# --------------------------------------------------------------------------- #
@dataclass
class HMMPosterior:
    """Posterior draws of the Gaussian-HMM parameters (``D`` draws, ``K`` states).

    States are ordered by volatility (``scales`` increasing): state 0 = calmest,
    state K-1 = most volatile. Ordering is enforced in the model, so it is stable
    across draws and interpretable.
    """

    probs_init: np.ndarray   # (D, K)     initial-state distribution
    probs_trans: np.ndarray  # (D, K, K)  row-stochastic transition matrix
    locs: np.ndarray         # (D, K)     per-state return mean
    scales: np.ndarray       # (D, K)     per-state return volatility
    n_states: int

    def mean_params(self):
        """Posterior-mean (probs_init, probs_trans, locs, scales)."""
        return (
            self.probs_init.mean(0), self.probs_trans.mean(0),
            self.locs.mean(0), self.scales.mean(0),
        )


@dataclass
class DriverSimulation:
    """Output of a driver fit+sample: ``(n_paths, horizon)`` per named driver."""

    paths: dict[str, np.ndarray]        # e.g. {"wti": (P, T)}
    seed: int
    posterior: HMMPosterior | None = None
    states: np.ndarray | None = None    # Viterbi most-likely regime path (history)
    returns: np.ndarray | None = field(default=None, repr=False)

    def quantiles(self, name: str, qs=(0.1, 0.5, 0.9)) -> np.ndarray:
        """Quantiles of a driver across paths, shape ``(len(qs), T)``."""
        return np.quantile(self.paths[name], qs, axis=0)


# --------------------------------------------------------------------------- #
# NumPyro model + NUTS fit
# --------------------------------------------------------------------------- #
def _hmm_model(returns, n_states):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from jax import lax
    from jax.scipy.special import logsumexp

    probs_init = numpyro.sample("probs_init", dist.Dirichlet(jnp.ones(n_states)))
    probs_trans = numpyro.sample(
        "probs_trans", dist.Dirichlet(jnp.ones((n_states, n_states))).to_event(1)
    )
    locs = numpyro.sample(
        "locs", dist.Normal(0.0, 0.05).expand([n_states]).to_event(1)
    )
    # Ordered log-scales break HMM label-switching → identifiable, stable regimes.
    log_scales = numpyro.sample(
        "log_scales",
        dist.TransformedDistribution(
            dist.Normal(jnp.log(0.05) * jnp.ones(n_states), 0.75).to_event(1),
            dist.transforms.OrderedTransform(),
        ),
    )
    scales = jnp.exp(log_scales)
    log_trans = jnp.log(probs_trans)

    def emit_logprob(y):  # (K,)
        return dist.Normal(locs, scales).log_prob(y)

    # Forward algorithm in log-space: α₀ then the α recursion over the series.
    log_alpha0 = jnp.log(probs_init) + emit_logprob(returns[0])

    def step(log_alpha, y):
        m = logsumexp(log_alpha[:, None] + log_trans, axis=0)  # (K,)
        return m + emit_logprob(y), None

    log_alpha_final, _ = lax.scan(step, log_alpha0, returns[1:])
    numpyro.factor("obs", logsumexp(log_alpha_final))  # marginal log-likelihood


def fit_hmm(
    returns,
    *,
    n_states: int = 2,
    num_warmup: int = 400,
    num_samples: int = 400,
    seed: int = 0,
    target_accept: float = 0.9,
) -> HMMPosterior:
    """Fit the Gaussian HMM to a 1-D array of log-returns via NUTS.

    Returns an :class:`HMMPosterior`. Discrete states are marginalized (forward
    algorithm), so NUTS samples only continuous parameters — no discrete-latent
    enumeration needed downstream.
    """
    import jax.numpy as jnp
    from jax import random
    from numpyro.infer import MCMC, NUTS

    returns = jnp.asarray(np.asarray(returns, dtype=float))
    kernel = NUTS(_hmm_model, target_accept_prob=target_accept)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    mcmc.run(random.PRNGKey(seed), returns=returns, n_states=n_states)
    s = mcmc.get_samples()
    return HMMPosterior(
        probs_init=np.asarray(s["probs_init"]),
        probs_trans=np.asarray(s["probs_trans"]),
        locs=np.asarray(s["locs"]),
        scales=np.asarray(jnp.exp(s["log_scales"])),
        n_states=n_states,
    )


# --------------------------------------------------------------------------- #
# Filtering / decoding (numpy; posterior-mean params)
# --------------------------------------------------------------------------- #
def _norm_logpdf(x, loc, scale):
    z = (x - loc) / scale
    return -0.5 * z * z - np.log(scale) - 0.5 * _LOG_2PI


def filtered_last_state(returns, post: HMMPosterior) -> np.ndarray:
    """P(s_T | y_1:T) at posterior-mean params — where the regime *is now*.

    Used to seed forward simulation from the current regime rather than the
    generic initial distribution.
    """
    pi, A, locs, scales = post.mean_params()
    log_alpha = np.log(pi) + _norm_logpdf(returns[0], locs, scales)
    logA = np.log(A)
    for y in returns[1:]:
        m = np.logaddexp.reduce(log_alpha[:, None] + logA, axis=0)
        log_alpha = m + _norm_logpdf(y, locs, scales)
    log_alpha -= log_alpha.max()
    w = np.exp(log_alpha)
    return w / w.sum()


def viterbi(returns, post: HMMPosterior) -> np.ndarray:
    """Most-likely regime sequence (Viterbi) at posterior-mean params."""
    pi, A, locs, scales = post.mean_params()
    logA = np.log(A)
    T, K = len(returns), post.n_states
    logdelta = np.log(pi) + _norm_logpdf(returns[0], locs, scales)
    psi = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        m = logdelta[:, None] + logA          # (K_from, K_to)
        psi[t] = m.argmax(axis=0)
        logdelta = m.max(axis=0) + _norm_logpdf(returns[t], locs, scales)
    states = np.zeros(T, dtype=int)
    states[-1] = int(logdelta.argmax())
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


# --------------------------------------------------------------------------- #
# Posterior-predictive forward simulation (numpy, vectorized over paths)
# --------------------------------------------------------------------------- #
def _categorical_rows(rng, probs):
    """Sample one category per row of ``probs`` (N, K) → (N,)."""
    c = np.cumsum(probs, axis=1)
    u = rng.random((probs.shape[0], 1))
    return (u > c).sum(axis=1).clip(max=probs.shape[1] - 1)


def simulate_from_posterior(
    post: HMMPosterior,
    last_value: float,
    *,
    n_paths: int,
    horizon: int,
    seed: int = 0,
    init_state_probs: np.ndarray | None = None,
) -> np.ndarray:
    """Posterior-predictive price paths, ``(n_paths, horizon)``.

    Each path draws a full parameter set from the posterior (capturing parameter
    uncertainty), seeds its regime from ``init_state_probs`` (defaults to the
    draw's own initial distribution), then rolls returns forward, compounding
    ``last_value`` into a price trajectory.
    """
    rng = np.random.default_rng(seed)
    d = post.locs.shape[0]
    idx = rng.integers(0, d, size=n_paths)
    pinit = post.probs_init[idx]     # (P, K)
    trans = post.probs_trans[idx]    # (P, K, K)
    locs = post.locs[idx]            # (P, K)
    scales = post.scales[idx]        # (P, K)

    if init_state_probs is not None:
        start = np.broadcast_to(init_state_probs, (n_paths, post.n_states))
        state = _categorical_rows(rng, start)
    else:
        state = _categorical_rows(rng, pinit)

    rows = np.arange(n_paths)
    price = np.full(n_paths, float(last_value))
    out = np.empty((n_paths, horizon))
    for t in range(horizon):
        ret = rng.normal(locs[rows, state], scales[rows, state])
        price = price * np.exp(ret)
        out[:, t] = price
        state = _categorical_rows(rng, trans[rows, state])
    return out


def save_posterior(post: HMMPosterior, path) -> None:
    """Persist a fitted posterior to ``.npz`` (a governance/reproducibility artifact)."""
    np.savez(
        path, probs_init=post.probs_init, probs_trans=post.probs_trans,
        locs=post.locs, scales=post.scales, n_states=post.n_states,
    )


def load_posterior(path) -> HMMPosterior:
    """Load a posterior saved by :func:`save_posterior`."""
    z = np.load(path)
    return HMMPosterior(
        probs_init=z["probs_init"], probs_trans=z["probs_trans"],
        locs=z["locs"], scales=z["scales"], n_states=int(z["n_states"]),
    )


def wti_scenario_paths(settings, history=None, *, refresh: bool = False):
    """Cached HMM WTI paths for the allocation pipeline → ``((S, T), wti0)``.

    Fits the HMM once and caches the posterior to a pinned-vintage ``.npz``, so
    repeated ``run_allocation`` calls only pay the fast simulation cost, not the
    NUTS fit. Returns the price paths and the reference (last-history) price.
    """
    from fce.drivers.history import load_wti_monthly

    if history is None:
        history = load_wti_monthly(settings)
    prices = np.asarray(history, dtype=float)
    returns = np.diff(np.log(prices))

    path = settings.data_dir / f"hmm_posterior.{settings.data_vintage}.npz"
    if path.exists() and not refresh:
        post = load_posterior(path)
    else:
        post = fit_hmm(returns, n_states=settings.hmm_states, seed=settings.seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_posterior(post, path)

    init = filtered_last_state(returns, post)
    paths = simulate_from_posterior(
        post, last_value=prices[-1], n_paths=settings.n_paths,
        horizon=settings.horizon_months, seed=settings.seed, init_state_probs=init,
    )
    return paths, float(prices[-1])


def simulate_drivers(
    history,
    *,
    n_paths: int,
    horizon: int,
    seed: int = 42,
    n_states: int = 2,
    **fit_kwargs,
) -> DriverSimulation:
    """Fit the HMM to a WTI price ``history`` and sample forward trajectories.

    ``history`` is a 1-D price series (monthly). Returns a
    :class:`DriverSimulation` with ``paths={"wti": (n_paths, horizon)}`` plus the
    fitted posterior and the Viterbi regime path for diagnostics/interpretability.
    """
    prices = np.asarray(history, dtype=float)
    returns = np.diff(np.log(prices))
    post = fit_hmm(returns, n_states=n_states, seed=seed, **fit_kwargs)
    init = filtered_last_state(returns, post)
    sim = simulate_from_posterior(
        post, last_value=prices[-1], n_paths=n_paths, horizon=horizon,
        seed=seed, init_state_probs=init,
    )
    states = viterbi(returns, post)
    logger.info(
        "HMM fit (%d states): posterior-mean vols %s, current-regime probs %s",
        n_states, np.round(post.scales.mean(0), 4), np.round(init, 3),
    )
    return DriverSimulation(
        paths={"wti": sim}, seed=seed, posterior=post, states=states, returns=returns,
    )
