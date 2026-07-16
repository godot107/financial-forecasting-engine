"""Tests for the Pillar-1 NumPyro Gaussian-HMM driver.

The headline test is **model recovery**: fit the HMM to data generated from a
*known* two-regime process and check the sampler recovers the regimes (ordered
volatilities, sticky calm state) and decodes the state path. One shared fit is
reused across tests (module fixture) so NUTS runs only once.
"""

from __future__ import annotations

import numpy as np
import pytest

from fce.config import Settings
from fce.drivers import (
    fit_hmm,
    simulate_from_posterior,
    simulate_true_hmm,
    viterbi,
)
from fce.drivers.history import _TRUE_SCALES, synthetic_wti_monthly
from fce.projects import simulate_project_scenarios


@pytest.fixture(scope="module")
def fitted():
    returns, states = simulate_true_hmm(400, seed=1)
    post = fit_hmm(returns, n_states=2, num_warmup=250, num_samples=250, seed=0)
    return returns, states, post


def test_hmm_recovers_ordered_regimes(fitted):
    _, _, post = fitted
    scales = post.scales.mean(0)
    # Ordered by construction: state 0 = calm, state 1 = crisis.
    assert scales[0] < scales[1]
    # Recovered vols land near the truth ([0.035, 0.130]).
    assert 0.02 < scales[0] < 0.06
    assert 0.09 < scales[1] < 0.18
    assert np.allclose(scales, _TRUE_SCALES, rtol=0.5)


def test_hmm_transition_matrix_is_stochastic_and_sticky(fitted):
    _, _, post = fitted
    A = post.probs_trans.mean(0)
    assert np.allclose(A.sum(axis=1), 1.0)          # rows are distributions
    assert A[0, 0] > 0.8                             # calm regime persists


def test_viterbi_decodes_the_state_path(fitted):
    returns, states, post = fitted
    decoded = viterbi(returns, post)
    accuracy = float(np.mean(decoded == states))
    assert accuracy > 0.75


def test_simulate_from_posterior_shapes_and_fan(fitted):
    _, _, post = fitted
    paths = simulate_from_posterior(post, last_value=70.0, n_paths=2000, horizon=24, seed=3)
    assert paths.shape == (2000, 24)
    assert np.all(paths > 0)                         # prices stay positive
    p10, p50, p90 = np.quantile(paths[:, -1], [0.1, 0.5, 0.9])
    assert p10 < p50 < p90                           # a real fan of outcomes


def test_hmm_paths_feed_the_allocation_scenarios():
    # End-to-end: HMM-driven WTI paths flow into the project scenario matrices.
    prices = synthetic_wti_monthly(240, seed=2)
    post = fit_hmm(np.diff(np.log(prices.values)), n_states=2,
                   num_warmup=150, num_samples=150, seed=0)
    wti = simulate_from_posterior(post, last_value=float(prices.iloc[-1]),
                                  n_paths=500, horizon=12, seed=0)
    scen = simulate_project_scenarios(Settings(n_paths=500, horizon_months=12),
                                      wti_paths=wti, wti0=float(prices.iloc[-1]))
    assert scen.npv_per_dollar.shape == (500, 5)
    assert np.isfinite(scen.npv_per_dollar).all()
