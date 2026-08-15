#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.final_opportunity.financial.resolver import FinancialResolver

def main():
    p = argparse.ArgumentParser(description="Check public financial-data availability.")
    p.add_argument("--ticker", required=True)
    p.add_argument("--cache-dir", default="data/financial_idx")
    args = p.parse_args()

    from app.final_opportunity.financial.idx_provider import IDXFinancialDataProvider
    resolver = FinancialResolver(IDXFinancialDataProvider(args.cache_dir))
    r = resolver.resolve(args.ticker, date.today())

    result = {
        "provider": resolver.provider.name,
        "ticker": args.ticker.upper().replace(".JK", ""),
        **r.to_dict(),
        "available": r.observation is not None,
        "source": resolver.provider.source_url,
    }
    print(json.dumps(result, indent=2))
    return 0 if r.observation else 1

if __name__ == "__main__":
    raise SystemExit(main())
