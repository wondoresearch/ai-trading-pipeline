from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PITObservation:
    ticker: str
    as_of: date
    publication_date: date
    price: float
    fundamental_score: float | None = None
    market_score: float | None = None
    news_score: float | None = None
    eligible: bool = True

    @property
    def point_in_time_valid(self) -> bool:
        return self.publication_date <= self.as_of


def _clip(value: float | None, lo: float = 0.0, hi: float = 1.0):
    if value is None:
        return None
    if not isfinite(float(value)):
        return None
    return max(lo, min(hi, float(value)))


def composite_score(
    row: PITObservation,
    *,
    market_weight: float = 0.40,
    news_weight: float = 0.25,
    fundamental_weight: float = 0.35,
) -> float | None:
    """Calculate a PIT-safe score using only information available by as_of."""
    if not row.eligible or not row.point_in_time_valid:
        return None

    weights = (market_weight, news_weight, fundamental_weight)
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative and have positive sum")

    values = {
        "market": _clip(row.market_score),
        "news": _clip(row.news_score),
        "fundamental": _clip(row.fundamental_score),
    }

    # Missing components are excluded and weights are renormalized.
    present = []
    for key, weight in zip(values, weights):
        if values[key] is not None:
            present.append((values[key], weight))

    if not present:
        return None

    total_weight = sum(w for _, w in present)
    return sum(v * w for v, w in present) / total_weight


def forward_return(
    prices: Mapping[str, float],
    ticker: str,
    horizon_prices: Mapping[str, float],
) -> float | None:
    """Simple forward return. No future price is used in score construction."""
    p0 = prices.get(ticker)
    p1 = horizon_prices.get(ticker)
    if p0 is None or p1 is None or p0 <= 0 or not isfinite(float(p0)) or not isfinite(float(p1)):
        return None
    return float(p1) / float(p0) - 1.0


def build_point_in_time_rows(
    observations: Iterable[PITObservation],
    *,
    market_weights=(0.40, 0.25, 0.35),
) -> list[dict]:
    """Build deterministic PIT rows and reject look-ahead observations."""
    out = []
    for obs in observations:
        score = composite_score(
            obs,
            market_weight=market_weights[0],
            news_weight=market_weights[1],
            fundamental_weight=market_weights[2],
        )
        if score is None:
            continue
        out.append({
            "ticker": obs.ticker.upper().replace(".JK", ""),
            "as_of": obs.as_of.isoformat(),
            "publication_date": obs.publication_date.isoformat(),
            "price": float(obs.price),
            "composite_score": round(score, 10),
        })
    return sorted(out, key=lambda x: (-x["composite_score"], x["ticker"]))


def rank_cross_section(rows: Sequence[Mapping]) -> list[dict]:
    """Stable rank: score descending, ticker ascending as deterministic tie-break."""
    ordered = sorted(rows, key=lambda r: (-float(r["composite_score"]), str(r["ticker"])))
    return [
        {**dict(row), "rank": i + 1}
        for i, row in enumerate(ordered)
    ]
