"""Sector-aware fundamental feature engine.

Pure functions only: no network calls. Accepts both the normalized
FinancialObservation field names used by the provider layer and the
semantic aliases used by older feature fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class FundamentalFeatures:
    ticker: str
    sector: str
    revenue_growth: Optional[float]
    net_income_growth: Optional[float]
    eps_growth: Optional[float]
    gross_margin: Optional[float]
    operating_margin: Optional[float]
    net_margin: Optional[float]
    profitability_score: float
    growth_score: float
    quality_score: float
    financial_score: float
    confidence: float
    trend: str
    sector_profile: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d


def _get(row: Any, name: str, aliases: tuple[str, ...] = ()) -> Optional[float]:
    for key in (name, *aliases):
        value = getattr(row, key, None)
        try:
            if value is None:
                continue
            value = float(value)
            return value
        except (TypeError, ValueError):
            continue
    return None


def _growth(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None or previous == 0:
        return None
    return current / previous - 1.0


def _margin(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(x) for x in values if x is not None]
    return sum(xs) / len(xs) if xs else None


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _growth_component(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return _clamp(0.5 + x / 0.4)


def _positive_component(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return _clamp(0.5 + x / 0.5)


def _sector_profile(sector: str, sub_industry: str = "") -> str:
    s = f"{sector} {sub_industry}".lower()
    if any(k in s for k in ("bank", "financial", "finansial")):
        return "banking"
    if any(k in s for k in ("coal", "mining", "mineral", "energy")):
        return "resources"
    if any(k in s for k in ("manufactur", "industrial", "consumer", "automotive")):
        return "industrial"
    return "general"


def build_fundamental_features(current: Any, previous: Any | None = None) -> FundamentalFeatures:
    sector = str(getattr(current, "sector", "") or "")
    sub_industry = str(getattr(current, "sub_industry", "") or "")
    profile = _sector_profile(sector, sub_industry)

    # Provider model aliases: Sales -> revenue, Profit for the Period /
    # attributable profit -> net income. This keeps the engine connected to
    # the actual IDX/Yahoo normalization contract instead of silently producing
    # empty growth features.
    revenue = _get(current, "revenue", ("sales",))
    prev_revenue = _get(previous, "revenue", ("sales",)) if previous else None
    net_income = _get(current, "net_income", ("profit_attributed", "profit"))
    prev_net_income = _get(previous, "net_income", ("profit_attributed", "profit")) if previous else None
    eps = _get(current, "eps")
    prev_eps = _get(previous, "eps") if previous else None

    gross_profit = _get(current, "gross_profit")
    operating_income = _get(current, "operating_income")

    revenue_growth = _growth(revenue, prev_revenue)
    net_income_growth = _growth(net_income, prev_net_income)
    eps_growth = _growth(eps, prev_eps)

    gross_margin = _margin(gross_profit, revenue)
    operating_margin = _margin(operating_income, revenue)
    net_margin = _margin(net_income, revenue)

    growth_score = _mean([
        _growth_component(revenue_growth),
        _growth_component(net_income_growth),
        _growth_component(eps_growth),
    ])
    profitability_score = _mean([
        _positive_component(gross_margin),
        _positive_component(operating_margin),
        _positive_component(net_margin),
    ])

    quality_inputs = []
    if profile == "banking":
        roe = _get(current, "roe")
        if roe is not None:
            quality_inputs.append(_positive_component(roe))
        pbv = _get(current, "pbv")
        if pbv is not None:
            quality_inputs.append(_clamp(1.0 - max(0.0, pbv - 1.0) / 4.0))
    else:
        debt_equity = _get(current, "debt_equity")
        if debt_equity is not None:
            quality_inputs.append(_clamp(1.0 - max(0.0, debt_equity - 0.5) / 3.0))
        roe = _get(current, "roe")
        if roe is not None:
            quality_inputs.append(_positive_component(roe))

    if profile == "banking":
        profitability_score = _mean([
            _growth_component(net_income_growth),
            _positive_component(_get(current, "roe")),
        ]) if any(x is not None for x in (net_income_growth, _get(current, "roe"))) else profitability_score

    quality_score = _mean(quality_inputs)
    if quality_score is None:
        quality_score = 0.5

    available = [revenue_growth, net_income_growth, eps_growth, gross_margin, operating_margin, net_margin]
    coverage = sum(x is not None for x in available) / len(available)
    confidence = _clamp(0.45 + coverage * 0.55)

    g = growth_score if growth_score is not None else 0.5
    p = profitability_score if profitability_score is not None else 0.5
    financial_score = _clamp(0.50 * g + 0.35 * p + 0.15 * quality_score)

    if financial_score >= 0.65:
        trend = "improving"
    elif financial_score <= 0.35:
        trend = "weakening"
    else:
        trend = "stable"

    reasons = []
    if net_income_growth is not None:
        reasons.append(f"net_income_growth={net_income_growth:.2%}")
    if revenue_growth is not None:
        reasons.append(f"revenue_growth={revenue_growth:.2%}")
    if eps_growth is not None:
        reasons.append(f"eps_growth={eps_growth:.2%}")
    if profile != "general":
        reasons.append(f"sector_profile={profile}")
    if not reasons:
        reasons.append("limited_fundamental_history")

    return FundamentalFeatures(
        ticker=str(getattr(current, "ticker", "") or ""),
        sector=sector,
        revenue_growth=revenue_growth,
        net_income_growth=net_income_growth,
        eps_growth=eps_growth,
        gross_margin=gross_margin,
        operating_margin=operating_margin,
        net_margin=net_margin,
        profitability_score=round(profitability_score if profitability_score is not None else 0.5, 6),
        growth_score=round(g, 6),
        quality_score=round(quality_score, 6),
        financial_score=round(financial_score, 6),
        confidence=round(confidence, 6),
        trend=trend,
        sector_profile=profile,
        reasons=tuple(reasons),
    )


def score_history(rows: list[Any]) -> FundamentalFeatures | None:
    if not rows:
        return None
    ordered = list(rows)
    return build_fundamental_features(ordered[0], ordered[1] if len(ordered) > 1 else None)
