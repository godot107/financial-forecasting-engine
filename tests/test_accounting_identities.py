"""Golden 3-statement articulation tests — the auditability guarantee.

These are the day-1 anchor: the whole project's selling point is a deterministic,
auditable accounting core, so we TDD the identities from the first commit. If any
of these break, the "GAAP-compliant / zero accounting drift" claim is void.
"""

from __future__ import annotations

import numpy as np
import pytest

from fce.accounting import assert_balanced, build_statements, npv
from fce.accounting.identities import AccountingIdentityError


def _random_drivers(seed=0, p=64, t=24):
    rng = np.random.default_rng(seed)
    revenue = rng.uniform(50e6, 150e6, size=(p, t))
    return dict(
        revenue=revenue,
        opex=0.5 * revenue,
        depreciation=np.full((p, t), 4e6),
        capex=rng.uniform(1e6, 6e6, size=(p, t)),
        delta_nwc=rng.normal(0, 2e6, size=(p, t)),
        interest=np.full((p, t), 1.5e6),
        net_debt_issued=rng.normal(0, 3e6, size=(p, t)),
    )


def _build(**overrides):
    drivers = _random_drivers()
    drivers.update(overrides)
    return build_statements(
        tax_rate=0.23,
        opening_cash=40e6,
        opening_ppe=250e6,
        opening_debt=100e6,
        opening_retained_earnings=20e6,
        **drivers,
    )


def test_balance_sheet_balances_every_path_and_period():
    stmt = _build()
    assert np.allclose(stmt.assets, stmt.liabilities_and_equity)


def test_assert_balanced_passes_on_articulated_statements():
    stmt = _build()
    # Must not raise.
    assert_balanced(stmt, opening_cash=40e6)


def test_cash_ties_to_cash_flow_statement():
    stmt = _build()
    reconstructed = 40e6 + np.cumsum(stmt.delta_cash, axis=1)
    assert np.allclose(stmt.cash, reconstructed)


def test_retained_earnings_roll_forward():
    stmt = _build()
    delta_re = np.diff(stmt.retained_earnings, axis=1)
    assert np.allclose(delta_re, stmt.net_income[:, 1:])


def test_dividends_reduce_retained_earnings():
    p, t = 64, 24
    div = np.full((p, t), 1e6)
    stmt = _build(net_debt_issued=np.zeros((p, t)))
    # Rebuild with dividends and check RE rolls with NI − div.
    drivers = _random_drivers()
    stmt2 = build_statements(
        tax_rate=0.23, dividends=div, opening_cash=40e6, opening_ppe=250e6,
        opening_debt=100e6, opening_retained_earnings=20e6, **drivers,
    )
    delta_re = np.diff(stmt2.retained_earnings, axis=1)
    assert np.allclose(delta_re, (stmt2.net_income - div)[:, 1:])
    assert_balanced(stmt2, opening_cash=40e6, dividends=div)


def test_assert_balanced_detects_corruption():
    stmt = _build()
    stmt.cash = stmt.cash + 1.0  # inject drift
    with pytest.raises(AccountingIdentityError):
        assert_balanced(stmt, opening_cash=40e6)


def test_npv_flat_rate_matches_manual_discount():
    p, t = 8, 12
    fcff = np.ones((p, t)) * 1e6
    rate = 0.12
    got = npv(fcff, rate)
    period_rate = rate / 12
    manual = sum(1e6 / (1 + period_rate) ** (j + 1) for j in range(t))
    assert np.allclose(got, manual)


def test_npv_zero_rate_is_plain_sum():
    fcff = np.arange(12, dtype=float).reshape(1, 12)
    assert np.isclose(npv(fcff, 0.0)[0], fcff.sum())
