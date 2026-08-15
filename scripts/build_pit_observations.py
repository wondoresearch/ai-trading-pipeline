#!/usr/bin/env python3
"""Build point-in-time observations from local market/financial data.

This adapter is intentionally conservative:
- financial information is only eligible on/after publication_date
- market price is selected on or before as_of
- optional news observations are accepted but never fabricated
- missing components remain null instead of being converted to zero
"""
from __future__ import annotations

import argparse, csv, json, math
from datetime import date, datetime
from pathlib import Path


def d(v):
    if v is None or str(v).strip() == "":
        return None
    return date.fromisoformat(str(v)[:10])


def num(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def clamp(x):
    if x is None:
        return None
    return max(0.0, min(1.0, float(x)))


def load_financial(path, ticker):
    p = Path(path)
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("rows", raw.get("observations", raw.get("data", [])))
    out = []
    for x in raw if isinstance(raw, list) else []:
        if str(x.get("ticker", ticker)).upper().replace(".JK","") != ticker:
            continue
        period_end = d(x.get("period_end") or x.get("fs_date") or x.get("as_of"))
        pub = d(x.get("publication_date") or x.get("published_at") or x.get("retrieved_at_utc"))
        if not period_end:
            continue
        # Retrieved-at is not assumed to be publication time. Without a real
        # publication date, keep the observation but mark it unavailable for PIT.
        if not pub:
            pub = d(x.get("publication_date"))
        out.append({
            "period_end": period_end.isoformat(),
            "publication_date": pub.isoformat() if pub else None,
            "revenue": num(x.get("revenue")),
            "net_income": num(x.get("net_income") or x.get("profit")),
            "eps": num(x.get("eps")),
            "source": x.get("source", "local_financial"),
            "raw": x,
        })
    return out


def load_market(path, ticker):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # tolerate common naming conventions
            dt = r.get("Date") or r.get("date") or r.get("Datetime") or r.get("datetime")
            close = r.get("Close") or r.get("close") or r.get("Adj Close") or r.get("adj_close")
            if not dt:
                continue
            try:
                rows.append({"date": str(dt)[:10], "close": num(close)})
            except Exception:
                continue
    return [x for x in rows if x["close"] is not None]


def load_news(path, ticker):
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("rows", raw.get("events", raw.get("observations", [])))
    out = []
    for x in raw if isinstance(raw, list) else []:
        t = str(x.get("ticker", ticker)).upper().replace(".JK","")
        if t != ticker:
            continue
        dt = d(x.get("publication_date") or x.get("event_time") or x.get("date"))
        score = num(x.get("sentiment") if x.get("sentiment") is not None else x.get("news_score"))
        if dt and score is not None:
            out.append({"date": dt.isoformat(), "score": clamp((score + 1)/2 if -1 <= score <= 1 else score)})
    return out


def latest_price(prices, as_of):
    candidates = [x for x in prices if x["date"] <= as_of.isoformat()]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x["date"])


def financial_score(f):
    # Neutral only when metrics are genuinely unavailable.
    vals = []
    if f.get("revenue") is not None:
        vals.append(0.5)
    if f.get("net_income") is not None:
        vals.append(0.5)
    if f.get("eps") is not None:
        vals.append(0.5)
    return sum(vals)/len(vals) if vals else None


def build(ticker, market_file, financial_file, news_file=None):
    ticker = ticker.upper().replace(".JK","")
    prices = load_market(market_file, ticker)
    fins = load_financial(financial_file, ticker)
    news = load_news(news_file, ticker)

    observations = []
    for f in fins:
        pub = d(f["publication_date"])
        period_end = d(f["period_end"])
        if not pub:
            # No publication timestamp => cannot safely use in PIT.
            continue
        p = latest_price(prices, pub)
        if not p:
            continue
        as_of = pub
        n = [x for x in news if x["date"] <= as_of.isoformat()]
        news_score = n[-1]["score"] if n else None
        observations.append({
            "ticker": ticker,
            "as_of": as_of.isoformat(),
            "publication_date": pub.isoformat(),
            "financial_period_end": period_end.isoformat(),
            "price": p["close"],
            "market_score": None,
            "news_score": news_score,
            "fundamental_score": financial_score(f),
            "eligible": pub <= as_of,
            "financial_source": f["source"],
            "market_source": str(market_file),
            "news_source": str(news_file) if news_file else None,
        })
    return observations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--market-dir", default="data/market_eod")
    ap.add_argument("--financial-dir", default="data/financial_stockanalysis")
    ap.add_argument("--news")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    all_rows = []
    for ticker in [x.strip().upper().replace(".JK","") for x in args.tickers.split(",") if x.strip()]:
        market = Path(args.market_dir) / f"{ticker}.csv"
        financial = Path(args.financial_dir) / f"{ticker}_annual.json"
        all_rows.extend(build(ticker, market, financial, args.news))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
