#!/usr/bin/env python3
"""Build deterministic PIT ranking dataset from normalized observations."""
from __future__ import annotations
import argparse, json
from datetime import date
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = []
    for x in raw:
        as_of = date.fromisoformat(x["as_of"])
        pub = date.fromisoformat(x["publication_date"])
        eligible = bool(x.get("eligible", True)) and pub <= as_of
        if not eligible:
            continue
        scores = [x.get(k) for k in ("market_score", "news_score", "fundamental_score")]
        scores = [float(v) for v in scores if v is not None]
        composite = sum(scores) / len(scores) if scores else None
        rows.append({
            **x,
            "composite_score": composite,
        })

    rows.sort(key=lambda x: (-x["composite_score"], x["ticker"])
              if x["composite_score"] is not None else (1e9, x["ticker"]))
    for i, x in enumerate(rows, 1):
        x["rank"] = i

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
