#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.final_opportunity.historical_ingest import import_source


def main():
    p = argparse.ArgumentParser(description="Import historical IDX EOD CSV data.")
    p.add_argument("--source", required=True, help="CSV file or directory of CSV files")
    p.add_argument("--output-dir", default="data/market_eod")
    p.add_argument("--tickers", nargs="*", help="Optional ticker allow-list")
    p.add_argument("--source-name", default="external_historical_dataset")
    p.add_argument("--source-url")
    args = p.parse_args()

    report = import_source(
        source=args.source,
        output_dir=args.output_dir,
        tickers=args.tickers,
        source_name=args.source_name,
        source_url=args.source_url,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
