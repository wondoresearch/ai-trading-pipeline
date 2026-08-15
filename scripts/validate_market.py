#!/usr/bin/env python3
import argparse
import json
from app.final_opportunity.market_validation import validate_directory

parser = argparse.ArgumentParser(description="Validate local IDX EOD CSV data.")
parser.add_argument("--data-dir", default="data/market_eod")
parser.add_argument("--json", action="store_true")
args = parser.parse_args()

report = validate_directory(args.data_dir)

if args.json:
    print(json.dumps(report, indent=2))
else:
    print("MARKET DATA COVERAGE")
    print("-" * 88)
    print(f"{'Ticker':8} {'Rows':>7} {'Start':12} {'End':12} {'Status':8}")
    for r in report["results"]:
        print(
            f"{r['ticker']:8} {r['rows']:7} "
            f"{str(r['first_date']):12} {str(r['last_date']):12} "
            f"{r['status']:8}"
        )
    print("-" * 88)
    print(
        f"Files: {report['files']}  Valid: {report['valid']}  "
        f"Invalid: {report['invalid']}  Coverage: {report['coverage_pct']}%"
    )

raise SystemExit(1 if report["invalid"] else 0)
