"""Point-in-time forward-return labels with strict trading-day semantics."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import csv


@dataclass(frozen=True)
class PricePoint:
    ticker: str
    day: date
    close: float


@dataclass(frozen=True)
class ForwardLabel:
    ticker: str
    as_of: date
    base_price: float
    horizon_days: int
    future_day: date | None
    future_price: float | None
    forward_return: float | None
    direction: int | None
    eligible: bool
    reason: str | None

    def to_dict(self):
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        if self.future_day is not None:
            d["future_day"] = self.future_day.isoformat()
        return d


def load_price_csv(path: str | Path, ticker: str | None = None) -> list[PricePoint]:
    path = Path(path)
    out: list[PricePoint] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = {str(x).strip().lower(): x for x in (reader.fieldnames or [])}
        date_col = next((fields[x] for x in ("date", "day", "datetime") if x in fields), None)
        close_col = next((fields[x] for x in ("close", "adj close", "adj_close") if x in fields), None)
        ticker_col = fields.get("ticker")
        if not date_col or not close_col:
            raise ValueError(f"Price CSV must contain date and close columns: {path}")
        for row in reader:
            t = ticker.upper() if ticker else (str(row[ticker_col]).upper() if ticker_col else path.stem.upper())
            try:
                day = date.fromisoformat(str(row[date_col])[:10])
                close = float(row[close_col])
            except (TypeError, ValueError):
                continue
            if close > 0:
                out.append(PricePoint(t, day, close))
    return sorted(out, key=lambda x: x.day)


def _future_point(points: Sequence[PricePoint], as_of: date, horizon: int) -> PricePoint | None:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    future = [p for p in points if p.day > as_of]
    if len(future) < horizon:
        return None
    return future[horizon - 1]


def make_label(observation: Mapping, points: Sequence[PricePoint], horizon: int) -> ForwardLabel:
    ticker = str(observation["ticker"]).upper()
    as_of = date.fromisoformat(str(observation["as_of"])[:10])
    base = float(observation["price"])
    if base <= 0:
        return ForwardLabel(ticker, as_of, base, horizon, None, None, None, None, False, "invalid_base_price")
    future = _future_point(points, as_of, horizon)
    if future is None:
        return ForwardLabel(ticker, as_of, base, horizon, None, None, None, None, False, "insufficient_future_prices")
    ret = future.close / base - 1.0
    direction = 1 if ret > 0 else (-1 if ret < 0 else 0)
    return ForwardLabel(ticker, as_of, base, horizon, future.day, future.close, ret, direction, True, None)


def build_forward_labels(observations: Iterable[Mapping], prices: Mapping[str, Sequence[PricePoint]], horizons=(1,3,5,10,20)) -> list[dict]:
    rows=[]
    for obs in observations:
        ticker=str(obs["ticker"]).upper()
        points=prices.get(ticker, ())
        for h in horizons:
            rows.append(make_label(obs, points, int(h)).to_dict())
    return rows


def audit_labels(rows: Iterable[Mapping]) -> dict:
    rows=list(rows); violations=[]; seen=set(); missing=0
    for r in rows:
        key=(r["ticker"], r["as_of"], int(r["horizon_days"]))
        if key in seen: violations.append(f"duplicate:{key}")
        seen.add(key)
        if r.get("eligible"):
            if r.get("future_day") is None or r.get("forward_return") is None:
                violations.append(f"eligible_without_future:{key}")
            if r["future_day"] <= r["as_of"]:
                violations.append(f"future_not_after_asof:{key}")
        else:
            missing += 1
    return {"status":"PASS" if not violations else "FAIL", "rows":len(rows), "eligible":sum(bool(r.get("eligible")) for r in rows), "excluded":missing, "violations":violations[:20]}
