"""Test the executive-summary reporter (Pillar 5)."""

from __future__ import annotations

import numpy as np

from fce.config import Settings
from fce.report import executive_summary


def test_executive_summary_is_coherent():
    summ = executive_summary(Settings(n_paths=4000, horizon_months=24))
    # Allocation respects the budget.
    assert summ.deployed <= summ.budget + 1.0
    assert np.all(summ.dollars >= -1e-6)
    # Scenario table has the baseline plus the named cases; baseline has no delta.
    assert summ.scenarios[0].name == "Baseline"
    assert summ.scenarios[0].d_npv == 0.0
    # Frontier is populated and the do-vs-condition overstatement is positive.
    assert len(summ.frontier) >= 5
    assert summ.do_overstatement_pct > 0
    # Markdown renders with the key sections.
    md = summ.to_markdown()
    assert "Recommendation" in md and "Scenario stress" in md and "Governance" in md
