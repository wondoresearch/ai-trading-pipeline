from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from math import isfinite
from statistics import mean
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BacktestObservation:
    ticker: str
    signal_date: date
    score: float
    forward_return: float
    feature_asof: date


@dataclass(frozen=True)
class BacktestMetrics:
    observations: int
    eligible_observations: int
    hit_rate: float | None
    average_forward_return: float | None
    top_bottom_spread: float | None
    information_coefficient: float | None

    def to_dict(self):
        return asdict(self)


def _safe_float(value: object) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if isfinite(x) else None


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 2 or len(a) != len(b):
        return None
    ma, mb = mean(a), mean(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    den_a = sum(x * x for x in da) ** 0.5
    den_b = sum(x * x for x in db) ** 0.5
    if den_a == 0 or den_b == 0:
        return None
    return sum(x * y for x, y in zip(da, db)) / (den_a * den_b)


def _validate_point_in_time(row: BacktestObservation) -> bool:
    # A feature is valid only when known no later than the signal date.
    return row.feature_asof <= row.signal_date


def evaluate(observations: Iterable[BacktestObservation], top_fraction: float = 0.20) -> BacktestMetrics:
    rows = list(observations)
    if not 0 < top_fraction <= 0.5:
        raise ValueError("top_fraction must be in (0, 0.5]")

    eligible = [
        r for r in rows
        if _safe_float(r.score) is not None
        and _safe_float(r.forward_return) is not None
        and _validate_point_in_time(r)
    ]
    if not eligible:
        return BacktestMetrics(len(rows), 0, None, None, None, None)

    scores = [float(r.score) for r in eligible]
    returns = [float(r.forward_return) for r in eligible]
    hit_rate = sum(r > 0 for r in returns) / len(returns)

    n_bucket = max(1, int(len(eligible) * top_fraction))
    ranked = sorted(eligible, key=lambda r: (-float(r.score), r.ticker))
    top = [float(r.forward_return) for r in ranked[:n_bucket]]
    bottom = [float(r.forward_return) for r in ranked[-n_bucket:]]

    return BacktestMetrics(
        observations=len(rows),
        eligible_observations=len(eligible),
        hit_rate=hit_rate,
        average_forward_return=mean(returns),
        top_bottom_spread=mean(top) - mean(bottom),
        information_coefficient=_pearson(_rank(scores), _rank(returns)),
    )
