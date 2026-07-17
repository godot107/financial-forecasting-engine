# Model-Validation Report

*Are the metrics trustworthy? — repricing, adequacy, and tail calibration.*

### Curve repricing round-trip

| Instrument | Tenor | Par quote | Repriced | Residual (bp) |
|---|---:|---:|---:|---:|
| deposit 0.25y | 0.25y | 5.250% | 5.250% | -0.000 |
| deposit 0.5y | 0.5y | 5.000% | 5.000% | -0.000 |
| deposit 1y | 1y | 4.700% | 4.700% | -0.000 |
| swap 2y | 2y | 4.300% | 4.300% | -0.000 |
| swap 3y | 3y | 4.100% | 4.100% | -0.000 |
| swap 5y | 5y | 4.000% | 4.000% | +0.000 |
| swap 7y | 7y | 4.100% | 4.100% | -0.000 |
| swap 10y | 10y | 4.200% | 4.200% | -0.000 |
| swap 30y | 30y | 4.400% | 4.400% | +0.000 |

**Max abs residual: 0.000 bp** (a healthy bootstrap reprices to well under 1 bp).

### HMM posterior predictive checks

Replicated series: 400. A p-value near 0.5 means the model reproduces the feature; near 0 or 1 flags misfit.

| Statistic | Observed | Replicated mean | PPC p-value |
|---|---:|---:|---:|
| Volatility | 0.0597 | 0.0602 | 0.458 |
| Excess kurtosis | 8.0960 | 5.8467 | 0.160 |
| Vol clustering (ACF₁ of r²) | 0.1450 | 0.1846 | 0.647 |
| Worst monthly return | -0.3233 | -0.2801 | 0.785 |

### VaR backtest (α = 95%)

- Observations: **419**
- Exceptions: **18** (observed 4.3% vs. expected 5.0%)
- Kupiec POF: LR = 0.46, p = 0.498 → PASS (unconditional coverage)
- Christoffersen CC: LR = 0.52, p = 0.770 → PASS (coverage + independence)

---

_Unit tests prove the code is correct; this report proves the numbers are calibrated. Regenerate: `python -m fce --validate`._
