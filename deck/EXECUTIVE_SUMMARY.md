# Executive Summary — Capital Allocation Recommendation

*Engine configuration: HMM drivers + QuantLib term structure + macro SCM. Budget $200M · CFaR₉₅ target $200M.*

## Recommendation

- **Expected portfolio NPV (P50): $150M**
- **Guaranteed liquidity (CFaR₉₅): $209M**
- Capital deployed: $200M of $200M

| Project | Allocation | % of budget |
|---|---:|---:|
| Alpha (Clean Retrofit) | $40.0M | 20.0% |
| Beta (LNG Terminal) | $90.0M | 45.0% |
| Gamma (Grid Battery) | $0.0M | 0.0% |
| Delta (Pipeline IoT) | $0.0M | 0.0% |
| Epsilon (Chemical Line) | $70.0M | 35.0% |

## Scenario stress (locked policy)

| Scenario | Expected NPV | Δ NPV | CFaR₉₅ | Breach |
|---|---:|---:|---:|:--:|
| Baseline | $150M | +0 | $209M |  |
| SOFR +200bp | $120M | -29 | $196M | ⚠️ |
| WTI −30% | $50M | -100 | $152M | ⚠️ |
| Rate spike + oil crash | $28M | -122 | $142M | ⚠️ |
| Soft landing (SOFR −100bp) | $165M | +15 | $215M |  |

## Risk–return frontier

- Buying liquidity is priced: moving CFaR₉₅ from $209M to $260M costs $41M of expected NPV.

## Governance

- Every scenario's balance sheet articulates (assets = liabilities + equity); accounting is deterministic and audited on every run.
- Scenario counterfactuals use *interventions* (`do`), not conditioning: conditioning would overstate the inflation→demand effect by ~109% here (confounding).
