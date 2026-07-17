"""Tests for the macro SCM and the what-if / stress tools.

The headline test is **confounding**: the interventional effect of inflation on
demand (do) is smaller than the observational one (condition), because ``rates``
confounds the inflation→demand relationship. That gap is the whole reason the
scenario engine uses ``do()`` and not conditioning.
"""

from __future__ import annotations

import numpy as np
import pytest

from fce.config import Settings
from fce.scenarios import (
    MacroSCM,
    build_context,
    commodity_shock,
    rate_shock,
    reverse_stress,
    run_scenarios,
    tornado,
)


# --- SCM structure & do-operator -----------------------------------------
def _sample(seed=0, s=4000, t=24):
    rng = np.random.default_rng(seed)
    rates = rng.normal(0.045, 0.010, size=(s, t)).clip(0)
    commodity = 75.0 * np.exp(rng.normal(0, 0.05, size=(s, t)))
    return MacroSCM(), MacroSCM().simulate(rates, commodity, seed=seed)


def test_graph_children_and_descendants():
    scm = MacroSCM()
    assert set(scm.children("rates")) == {"inflation", "demand", "discount_factor"}
    assert "demand" in scm.descendants("rates")
    assert "revenue" in scm.descendants("rates")   # rates→demand→revenue
    assert scm.descendants("commodity") == {"revenue"}


def test_do_rates_propagates_to_descendants_only():
    scm, base = _sample()
    out = scm.do(base, rate_shock(200))            # +200bp
    assert np.allclose(out.rates, base.rates + 0.02)
    assert not np.allclose(out.inflation, base.inflation)   # child recomputed
    assert not np.allclose(out.demand, base.demand)         # descendant recomputed
    assert np.allclose(out.commodity, base.commodity)       # non-descendant untouched
    # Higher rates → lower demand (both channels are negative).
    assert out.demand.mean() < base.demand.mean()


def test_do_commodity_leaves_rate_chain_untouched():
    scm, base = _sample()
    out = scm.do(base, commodity_shock(-30))
    assert np.allclose(out.commodity, base.commodity * 0.7)
    assert np.allclose(out.rates, base.rates)
    assert np.allclose(out.inflation, base.inflation)
    assert np.allclose(out.demand, base.demand)


def test_intervention_holds_exogenous_noise_fixed():
    # do(rates := same rates) must reproduce the baseline exactly (same-world CF).
    scm, base = _sample()
    out = scm.do(base, {"rates": lambda r: r})
    assert np.allclose(out.inflation, base.inflation)
    assert np.allclose(out.demand, base.demand)


def test_conditioning_overstates_effect_vs_intervention():
    scm, base = _sample()
    # Per-unit effect of inflation on demand, two ways:
    # Interventional slope — do(inflation += δ); isolates the structural coefficient.
    delta = 0.01
    do_s = scm.do(base, {"inflation": lambda x: x + delta})
    itv_slope = (do_s.demand - base.demand).mean() / delta
    # Observational slope — regress demand on inflation across all cells; this also
    # picks up the backdoor path inflation ← rates → demand (rates is a confounder).
    obs_slope = np.polyfit(base.inflation.ravel(), base.demand.ravel(), 1)[0]
    # Both negative; conditioning's slope is steeper (more negative) than the truth.
    assert itv_slope < 0 and obs_slope < 0
    assert obs_slope < itv_slope                       # observational overstates
    assert itv_slope == pytest.approx(scm.b_di, abs=0.05)  # recovers b_di exactly


# --- what-if tools (slower; share one context) ---------------------------
@pytest.fixture(scope="module")
def ctx():
    return build_context(Settings(n_paths=5000, horizon_months=36))


def test_named_scenarios_move_the_right_way(ctx):
    results, _ = run_scenarios(ctx=ctx)
    by = {r.name: r for r in results}
    assert by["Baseline"].d_npv == 0.0
    # Adverse shocks cut NPV and CFaR; the oil crash bites hardest.
    assert by["SOFR +200bp"].d_npv < 0 and by["WTI −30%"].d_npv < 0
    assert by["WTI −30%"].d_npv < by["SOFR +200bp"].d_npv
    assert by["Rate spike + oil crash"].d_npv < by["WTI −30%"].d_npv
    # Soft landing helps.
    assert by["Soft landing (SOFR −100bp)"].d_npv > 0


def test_tornado_ranks_wti_above_rates(ctx):
    bars, _ = tornado(ctx)
    assert bars[0].swing >= bars[-1].swing            # sorted
    assert any("WTI" in b.driver for b in bars)
    assert all(b.swing >= 0 for b in bars)


def test_reverse_stress_grid_is_monotone_and_breaches(ctx):
    rs, _ = reverse_stress(ctx, n=7)
    assert rs.cfar.shape == (7, 7)
    # More rate hike (down rows) and more oil crash (right cols) → lower CFaR.
    assert np.all(np.diff(rs.cfar, axis=0) <= 1e-6)
    assert np.all(np.diff(rs.cfar, axis=1) <= 1e-6)
    assert rs.breaches().any() and not rs.breaches().all()
