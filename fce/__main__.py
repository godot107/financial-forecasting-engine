"""CLI entrypoint: ``python -m fce``.

Thin wrapper over :func:`fce.pipeline.run_pipeline` — no logic here.
"""

from __future__ import annotations

import argparse
import logging

from fce.pipeline import run_allocation, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="fce", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument(
        "--allocate", action="store_true",
        help="run the full CFaR-constrained capital allocation (Pillars 3+4)",
    )
    parser.add_argument(
        "--hmm", action="store_true",
        help="drive allocation with the Pillar-1 NumPyro HMM (else placeholder GBM)",
    )
    parser.add_argument(
        "--quantlib", action="store_true",
        help="add Pillar-2 term structure: floating-rate debt + curve discounting",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="print the CFO-ready executive summary (Markdown)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.report:
        from fce.report import executive_summary

        print(executive_summary(use_hmm=args.hmm).to_markdown())
        return

    if args.allocate:
        _print_allocation(run_allocation(use_hmm=args.hmm, use_quantlib=args.quantlib))
        return

    result = run_pipeline()
    print("\nMVP capital-forecasting slice (placeholder driver):")
    print(f"  paths          : {result.n_paths:,}  ×  {result.horizon_months} months")
    print(f"  balanced       : {result.balanced}")
    print(f"  project NPV P10 : ${result.npv_p10/1e6:,.1f}M")
    print(f"  project NPV P50 : ${result.npv_p50/1e6:,.1f}M")
    print(f"  project NPV P90 : ${result.npv_p90/1e6:,.1f}M")


def _print_allocation(res) -> None:
    a = res.allocation
    print(f"\nCapital allocation — ${res.budget/1e6:,.0f}M budget, "
          f"CFaR₉₅ floor ${res.cfar_floor/1e6:,.0f}M  [{a.status}]")
    print(f"  Expected portfolio NPV : ${a.expected_npv/1e6:,.1f}M")
    print(f"  Guaranteed liquidity (CFaR₉₅): ${a.cfar/1e6:,.1f}M")
    print(f"  Deployed               : ${a.deployed/1e6:,.0f}M")
    print("  Recommended allocation:")
    for name, dollars, w in zip(a.names, a.dollars, a.weights):
        bar = "█" * round(w * 25)
        print(f"    {name:26s} ${dollars/1e6:6.1f}M  {w*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
