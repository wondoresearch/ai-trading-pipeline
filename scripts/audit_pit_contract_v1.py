#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import Counter
from datetime import date
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload

    keys = [(r.get("ticker"), r.get("as_of")) for r in rows]
    counts = Counter(keys)
    duplicates = [k for k, n in counts.items() if n > 1]

    violations = []
    for r in rows:
        try:
            pub = date.fromisoformat(r["publication_date"])
            as_of = date.fromisoformat(r["as_of"])
            fin = date.fromisoformat(r["financial_period_end"])
        except Exception:
            violations.append((r.get("ticker"), "invalid_or_missing_date"))
            continue

        if pub > as_of:
            violations.append((r["ticker"], "publication_after_as_of"))
        if fin > pub:
            violations.append((r["ticker"], "financial_period_after_publication"))

        if r.get("pit_valid") is not True:
            # Current snapshots are allowed to exist, but are explicitly not
            # eligible for historical PIT/backtesting.
            pass

    status = "PASS" if not duplicates and not violations else "FAIL"
    result = {
        "status": status,
        "rows": len(rows),
        "duplicates": duplicates,
        "date_violations": violations,
        "historical_pit_eligible_rows": sum(1 for r in rows if r.get("pit_valid") is True),
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)

if __name__ == "__main__":
    main()
