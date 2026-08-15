#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.final_opportunity.financial.stockanalysis_provider import StockAnalysisFinancialProvider


def main():
    parser = argparse.ArgumentParser(description="Check public StockAnalysis financial data.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--quarterly", action="store_true")
    parser.add_argument("--cache-dir", default="data/financial_stockanalysis")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    provider = StockAnalysisFinancialProvider(args.cache_dir)
    try:
        rows = provider.fetch(args.ticker, quarterly=args.quarterly)
    except Exception as exc:
        result = {
            "ticker": args.ticker.upper().replace(".JK", ""),
            "provider": provider.name,
            "status": "error",
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(result, indent=2))
        return 1

    result = {
        "ticker": args.ticker.upper().replace(".JK", ""),
        "provider": provider.name,
        "status": "available" if rows else "not_found",
        "available": bool(rows),
        "period_type": "quarterly" if args.quarterly else "annual",
        "rows": len(rows),
        "source": provider.url(args.ticker, args.quarterly),
        "observation": rows[0].to_dict() if rows else None,
    }
    print(json.dumps(result, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
