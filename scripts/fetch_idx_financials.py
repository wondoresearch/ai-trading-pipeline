#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.final_opportunity.financial.idx_provider import IDXFinancialDataProvider


def main():
    p = argparse.ArgumentParser(description="Fetch free public IDX financial data.")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--tickers", nargs="*", default=None)
    p.add_argument("--cache-dir", default="data/financial_idx")
    args = p.parse_args()

    provider = IDXFinancialDataProvider(args.cache_dir)
    rows = provider.fetch_month(args.year, args.month, args.tickers)
    print(json.dumps([r.to_dict() for r in rows], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
