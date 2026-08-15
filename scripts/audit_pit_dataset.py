#!/usr/bin/env python3
"""Audit PIT dataset for temporal leakage and structural quality."""
from __future__ import annotations
import argparse, json
from datetime import date
from collections import Counter


def dd(v):
    return date.fromisoformat(str(v)[:10])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    args = ap.parse_args()
    rows = json.loads(open(args.input, encoding="utf-8").read())

    violations = []
    duplicates = []
    seen = set()
    sectors = Counter()
    eligible = 0

    for i, r in enumerate(rows):
        key = (r.get("ticker"), r.get("as_of"), r.get("publication_date"))
        if key in seen:
            duplicates.append(key)
        seen.add(key)

        try:
            as_of = dd(r["as_of"])
            pub = dd(r["publication_date"])
            if pub > as_of:
                violations.append((i, "publication_after_as_of"))
        except Exception:
            violations.append((i, "invalid_date"))

        if r.get("eligible"):
            eligible += 1
        if r.get("sector"):
            sectors[r["sector"]] += 1

    status = "PASS" if not violations and not duplicates else "FAIL"
    result = {
        "status": status,
        "rows": len(rows),
        "eligible": eligible,
        "excluded": len(rows) - eligible,
        "lookahead_violations": len(violations),
        "duplicate_observations": len(duplicates),
        "sector_coverage": len(sectors),
        "violations_sample": violations[:10],
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
