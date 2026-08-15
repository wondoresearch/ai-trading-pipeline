#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.final_opportunity.financial.composite_provider import CompositeFinancialProvider


def main():
    p = argparse.ArgumentParser(
        description="Check free financial-provider availability and provenance."
    )
    p.add_argument("--ticker", required=True)
    p.add_argument("--quarterly", action="store_true")
    args = p.parse_args()

    provider = CompositeFinancialProvider()
    result = provider.status(args.ticker, quarterly=args.quarterly)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
