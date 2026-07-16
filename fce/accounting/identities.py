"""Golden accounting-identity checks — the auditability guarantee, as code.

``assert_balanced`` is the single gate every simulated state must pass. It is
deliberately cheap so it can run inside the hot Monte Carlo loop as a debug
assertion, and it is the spec the golden tests exercise. Three articulation
identities, all of which must hold for every path and every period:

1. **Balance sheet balances:**  Assets == Liabilities + Equity.
2. **Cash ties out:**           end-of-period cash == opening cash + ΣΔcash.
3. **Retained earnings roll:**  ΔRE == Net Income − Dividends.
"""

from __future__ import annotations

import numpy as np

from fce.accounting.statements import Statements


class AccountingIdentityError(AssertionError):
    """Raised when a statement articulation fails to balance."""


def assert_balanced(
    stmt: Statements,
    *,
    opening_cash: np.ndarray | float = 0.0,
    dividends: np.ndarray | float | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-9,
) -> None:
    """Assert the three articulation identities hold for all paths/periods.

    ``atol``/``rtol`` are scaled tolerances for float roundoff — corporate
    balances can be large, so relative tolerance matters. Raises
    :class:`AccountingIdentityError` naming the first identity that fails.
    """
    # 1. Balance sheet balances. Compare the two sides directly (not diff vs 0)
    # so ``rtol`` scales with the balance magnitude — corporate balances are
    # ~1e9, where 1e-6 absolute roundoff is negligible but would trip an
    # atol-only check against zero.
    if not np.allclose(stmt.assets, stmt.liabilities_and_equity, atol=atol, rtol=rtol):
        worst = float(np.max(np.abs(stmt.assets - stmt.liabilities_and_equity)))
        raise AccountingIdentityError(
            f"Balance sheet does not balance: max |A − (L+E)| = {worst:.6g}"
        )

    # 2. Cash ties to the cash-flow statement.
    p = stmt.cash.shape[0]
    open_cash = np.broadcast_to(np.asarray(opening_cash, float), (p,)).reshape(p, 1)
    reconstructed_cash = open_cash + np.cumsum(stmt.delta_cash, axis=1)
    if not np.allclose(stmt.cash, reconstructed_cash, atol=atol, rtol=rtol):
        worst = float(np.max(np.abs(stmt.cash - reconstructed_cash)))
        raise AccountingIdentityError(
            f"Cash does not tie to the cash-flow statement: max diff = {worst:.6g}"
        )

    # 3. Retained-earnings roll-forward.
    div = (
        np.zeros_like(stmt.net_income)
        if dividends is None
        else np.broadcast_to(np.asarray(dividends, float), stmt.net_income.shape)
    )
    delta_re = np.diff(stmt.retained_earnings, axis=1)
    expected = (stmt.net_income - div)[:, 1:]
    if not np.allclose(delta_re, expected, atol=atol, rtol=rtol):
        worst = float(np.max(np.abs(delta_re - expected)))
        raise AccountingIdentityError(
            f"Retained earnings do not roll forward: max diff = {worst:.6g}"
        )
