"""Pillar 1 — probabilistic macro driver engine (NumPyro).

Bayesian structural time-series + Hidden Markov / state-space models fit to the
real observable drivers (WTI, Treasury yields, inflation), then sampled forward
into thousands of multi-horizon Monte Carlo trajectories with honest P10/P50/P90
distributions.

**Locked decision #2: NumPyro only** — it rides the JAX already needed for the
accounting layer. Stan is intentionally dropped (C++ toolchain, no portfolio
payoff); keep "Stan-ready" as a talking point only.

Reference implementation: Dixon, Halperin & Bilokon, *Machine Learning in
Finance*, Ch. 7 (HMM transition/emission factorization; Kalman state-space
fitting). Reinforced by Nielsen, *Practical Time Series Analysis* (filtering /
smoothing / Viterbi).

Built: :func:`fit_hmm` (NUTS + forward algorithm), :func:`simulate_drivers`
(fit → posterior-predictive forward sim), plus :func:`viterbi` decoding and
:func:`load_wti_monthly` (real EIA or offline synthetic history).
"""

from fce.drivers.history import (
    load_wti_monthly,
    simulate_true_hmm,
    synthetic_wti_monthly,
)
from fce.drivers.hmm import (
    DriverSimulation,
    HMMPosterior,
    fit_hmm,
    load_posterior,
    save_posterior,
    simulate_drivers,
    simulate_from_posterior,
    viterbi,
    wti_scenario_paths,
)

__all__ = [
    "DriverSimulation", "HMMPosterior", "fit_hmm", "simulate_drivers",
    "simulate_from_posterior", "viterbi", "wti_scenario_paths",
    "save_posterior", "load_posterior",
    "load_wti_monthly", "synthetic_wti_monthly", "simulate_true_hmm",
]
