#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from datetime import date
from pathlib import Path

def d(x):
    return date.fromisoformat(x) if x else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--as-of", required=True)
    args = ap.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    as_of = date.fromisoformat(args.as_of)

    by_ticker = {}
    rejected = []

    for r in raw:
        ticker = str(r["ticker"]).upper().replace(".JK", "")
        row_as_of = d(r.get("as_of"))
        if row_as_of != as_of:
            rejected.append({"ticker": ticker, "reason": "different_as_of"})
            continue

        # A publication date equal to the snapshot date is not proof that the
        # financial period was published on that date. It is accepted only for
        # a current snapshot, never as historical evidence.
        fin_end = d(r.get("financial_period_end"))
        if fin_end is None:
            rejected.append({"ticker": ticker, "reason": "missing_financial_period_end"})
            continue

        old = by_ticker.get(ticker)
        if old is None or d(old["financial_period_end"]) < fin_end:
            by_ticker[ticker] = dict(r)

    out = []
    for ticker, r in sorted(by_ticker.items()):
        r["ticker"] = ticker
        r["pit_valid"] = False
        r["pit_valid_reason"] = (
            "current_snapshot_only: publication date is not independently "
            "verified for the financial period"
        )
        out.append(r)

    payload = {
        "schema": "pit_reconstruction_v1",
        "as_of": args.as_of,
        "rows": out,
        "rejected_rows": rejected,
        "historical_pit_ready": False,
        "reason": "Historical publication dates are not proven by the current source cache."
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "rows": len(out),
        "rejected": len(rejected),
        "historical_pit_ready": False,
        "output": args.output
    }, indent=2))

if __name__ == "__main__":
    main()
