"""Vectorized 3-statement articulation and NPV.

Every array is shaped ``(P, T)`` — ``P`` Monte Carlo paths × ``T`` periods —
except opening balances, which are ``(P,)`` (or scalar, broadcast). The engine
is a plain roll-forward that balances *by construction*: see the module docstring
of :mod:`fce.accounting` and the derivation in ``tests/test_accounting_identities.py``.

MVP scope (locked decision #1): one project, one driver (WTI → revenue → FCFF →
NPV). Multi-project breadth is a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Statements:
    """Fully articulated statements for ``P`` paths over ``T`` periods.

    Income statement, cash-flow statement, and balance-sheet stocks are each
    ``(P, T)`` arrays. Balance-sheet fields are *end-of-period* balances.
    """

    # Income statement
    revenue: np.ndarray
    ebitda: np.ndarray
    ebit: np.ndarray
    interest: np.ndarray
    tax: np.ndarray
    net_income: np.ndarray
    # Cash flow
    cfo: np.ndarray
    cfi: np.ndarray
    cff: np.ndarray
    delta_cash: np.ndarray
    fcff: np.ndarray
    # Balance sheet (end-of-period stocks)
    cash: np.ndarray
    nwc: np.ndarray
    ppe: np.ndarray
    debt: np.ndarray
    common_stock: np.ndarray
    retained_earnings: np.ndarray

    @property
    def assets(self) -> np.ndarray:
        return self.cash + self.nwc + self.ppe

    @property
    def liabilities_and_equity(self) -> np.ndarray:
        return self.debt + self.common_stock + self.retained_earnings


def build_statements(
    *,
    revenue: np.ndarray,
    opex: np.ndarray,
    depreciation: np.ndarray,
    capex: np.ndarray,
    delta_nwc: np.ndarray,
    interest: np.ndarray,
    net_debt_issued: np.ndarray,
    tax_rate: float,
    dividends: np.ndarray | None = None,
    opening_cash: np.ndarray | float = 0.0,
    opening_nwc: np.ndarray | float = 0.0,
    opening_ppe: np.ndarray | float = 0.0,
    opening_debt: np.ndarray | float = 0.0,
    opening_retained_earnings: np.ndarray | float = 0.0,
) -> Statements:
    """Roll forward a balanced 3-statement model from driver vectors.

    All flow inputs are ``(P, T)``. Opening balances are ``(P,)`` or scalar.
    ``common_stock`` is solved once so the *opening* balance sheet balances, then
    held constant — every subsequent period balances by construction because
    ΔAssets = ΔEquity + ΔDebt each step (see the identity test).

    Taxes use a simple ``tax_rate × max(EBT, 0)`` rule (no loss carryforward in
    the MVP). Returns a :class:`Statements` with all articulations populated.
    """
    revenue = np.asarray(revenue, dtype=float)
    p, t = revenue.shape
    dividends = np.zeros((p, t)) if dividends is None else np.asarray(dividends, float)

    def _flow(x) -> np.ndarray:
        return np.broadcast_to(np.asarray(x, float), (p, t)).copy()

    opex = _flow(opex)
    depreciation = _flow(depreciation)
    capex = _flow(capex)
    delta_nwc = _flow(delta_nwc)
    interest = _flow(interest)
    net_debt_issued = _flow(net_debt_issued)

    # --- Income statement -------------------------------------------------
    ebitda = revenue - opex
    ebit = ebitda - depreciation
    ebt = ebit - interest
    tax = tax_rate * np.maximum(ebt, 0.0)
    net_income = ebt - tax

    # --- Cash flow --------------------------------------------------------
    cfo = net_income + depreciation - delta_nwc
    cfi = -capex
    cff = net_debt_issued - dividends
    delta_cash = cfo + cfi + cff

    # Unlevered free cash flow to the firm (what the optimizer discounts).
    fcff = ebit * (1.0 - tax_rate) + depreciation - capex - delta_nwc

    # --- Balance-sheet roll-forward (cumulative sums over the period axis) -
    def _open(x) -> np.ndarray:
        return np.broadcast_to(np.asarray(x, float), (p,)).reshape(p, 1)

    cash = _open(opening_cash) + np.cumsum(delta_cash, axis=1)
    nwc = _open(opening_nwc) + np.cumsum(delta_nwc, axis=1)
    ppe = _open(opening_ppe) + np.cumsum(capex - depreciation, axis=1)
    debt = _open(opening_debt) + np.cumsum(net_debt_issued, axis=1)
    retained_earnings = _open(opening_retained_earnings) + np.cumsum(
        net_income - dividends, axis=1
    )

    # Solve common stock so the OPENING balance sheet balances; hold constant.
    #   opening_assets = opening_debt + common_stock + opening_RE
    opening_assets = (
        _open(opening_cash) + _open(opening_nwc) + _open(opening_ppe)
    ).reshape(p)
    common_stock_vec = (
        opening_assets
        - np.broadcast_to(np.asarray(opening_debt, float), (p,))
        - np.broadcast_to(np.asarray(opening_retained_earnings, float), (p,))
    )
    common_stock = np.broadcast_to(common_stock_vec.reshape(p, 1), (p, t)).copy()

    return Statements(
        revenue=revenue,
        ebitda=ebitda,
        ebit=ebit,
        interest=interest,
        tax=tax,
        net_income=net_income,
        cfo=cfo,
        cfi=cfi,
        cff=cff,
        delta_cash=delta_cash,
        fcff=fcff,
        cash=cash,
        nwc=nwc,
        ppe=ppe,
        debt=debt,
        common_stock=common_stock,
        retained_earnings=retained_earnings,
    )


def npv(fcff: np.ndarray, rate: np.ndarray | float, *, periods_per_year: int = 12) -> np.ndarray:
    """Discount an ``(P, T)`` FCFF matrix to a ``(P,)`` NPV vector.

    ``rate`` is an *annual* discount rate — scalar (flat WACC) or ``(P, T)``
    (per-path/per-period discount from the QuantLib term structure). Period-end
    convention: the ``j``-th column is discounted by ``(1 + rate/ppy)**(j+1)``.
    """
    fcff = np.asarray(fcff, dtype=float)
    p, t = fcff.shape
    period_rate = np.asarray(rate, float) / periods_per_year
    exponents = np.arange(1, t + 1).reshape(1, t)
    discount = (1.0 + period_rate) ** exponents
    return np.sum(fcff / discount, axis=1)


def present_value(fcff: np.ndarray, discount_factors: np.ndarray) -> np.ndarray:
    """Discount an ``(P, T)`` FCFF matrix with explicit discount factors → ``(P,)``.

    ``discount_factors`` is ``(P, T)`` (per-path/per-period, e.g. from the Pillar-2
    term structure) or ``(T,)`` (one curve for all paths). This is the general
    form of :func:`npv`; use it when discounting comes from a bootstrapped/simulated
    curve rather than a flat rate.
    """
    fcff = np.asarray(fcff, dtype=float)
    df = np.asarray(discount_factors, dtype=float)
    if df.ndim == 1:
        df = df.reshape(1, -1)
    return np.sum(fcff * df, axis=1)
