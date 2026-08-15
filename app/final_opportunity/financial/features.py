"""Sector-neutral fundamental feature engineering.

The key design rule is: never compare a bank with an industrial/mining/
consumer company using a bank-only metric set. The provider's IDX sector and
sub-industry labels determine the feature family.
"""

from __future__ import annotations

from typing import Iterable, Optional
from .models import FinancialObservation, FinancialFeatureSet


def classify_sector(sector: str, sub_industry: str = "") -> str:
    text = f"{sector} {sub_industry}".lower()
    if any(x in text for x in ["financials", "bank", "banking", "insurance", "finance"]):
        return "financial"
    if any(x in text for x in ["energy", "coal", "oil", "gas", "mining", "mineral"]):
        return "resource"
    if any(x in text for x in ["industrial", "manufactur", "automotive", "chemical", "machinery"]):
        return "industrial"
    if any(x in text for x in ["consumer", "retail", "food", "beverage", "household"]):
        return "consumer"
    if any(x in text for x in ["technology", "software", "telecommunication", "digital"]):
        return "technology"
    if any(x in text for x in ["property", "real estate", "reit"]):
        return "property"
    if any(x in text for x in ["infrastructure", "transport", "logistics", "utility"]):
        return "infrastructure"
    if any(x in text for x in ["health", "pharma", "hospital"]):
        return "healthcare"
    return "other"


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _positive_signal(value: Optional[float], good: float, bad: float) -> Optional[float]:
    if value is None:
        return None
    if value >= good:
        return 1.0
    if value <= bad:
        return 0.0
    return (value - bad) / (good - bad)


def _negative_signal(value: Optional[float], good: float, bad: float) -> Optional[float]:
    if value is None:
        return None
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return (bad - value) / (bad - good)


def build_features(
    observations: Iterable[FinancialObservation],
    previous: Optional[FinancialObservation] = None,
) -> FinancialFeatureSet:
    obs = list(observations)[0]
    group = classify_sector(obs.sector, obs.sub_industry)
    reasons = []

    profitability_parts = [
        x for x in (
            _positive_signal(obs.roe, 15, 0),
            _positive_signal(obs.roa, 5, 0),
            _positive_signal(obs.npm, 15, 0),
        ) if x is not None
    ]
    profitability = sum(profitability_parts) / len(profitability_parts) if profitability_parts else None

    growth = None
    if previous:
        growth_parts = []
        if obs.sales is not None and previous.sales not in (None, 0):
            growth_parts.append(_clamp((obs.sales / previous.sales) - 0.8, 0, 1) / 0.4)
        if obs.profit_attributed is not None and previous.profit_attributed not in (None, 0):
            growth_parts.append(_clamp((obs.profit_attributed / previous.profit_attributed) - 0.7, 0, 1) / 0.6)
        if growth_parts:
            growth = sum(growth_parts) / len(growth_parts)

    leverage = _negative_signal(obs.debt_equity, 0.5, 3.0)
    valuation = None
    if obs.pe is not None and obs.pe > 0:
        valuation = _negative_signal(obs.pe, 10, 35)
    if obs.pbv is not None and obs.pbv > 0:
        pbv_signal = _negative_signal(obs.pbv, 1.5, 6)
        valuation = pbv_signal if valuation is None else (valuation + pbv_signal) / 2

    parts = [x for x in (profitability, growth, leverage, valuation) if x is not None]
    quality = len(parts) / 4.0
    score = sum(parts) / len(parts) if parts else 0.0

    if profitability is not None:
        reasons.append(f"profitability={profitability:.2f}")
    if growth is not None:
        reasons.append(f"growth={growth:.2f}")
    if leverage is not None:
        reasons.append(f"leverage={leverage:.2f}")
    if valuation is not None:
        reasons.append(f"valuation={valuation:.2f}")

    # Financial-sector leverage is not interpreted like industrial leverage.
    if group == "financial":
        reasons.append("financial-sector profile: leverage is treated as contextual, not a hard debt-risk proxy")
        score = sum(x for x in (profitability, growth, valuation) if x is not None)
        denom = len([x for x in (profitability, growth, valuation) if x is not None])
        score = score / denom if denom else 0.0

    return FinancialFeatureSet(
        ticker=obs.ticker,
        sector_group=group,
        observation_date=obs.fs_date,
        quality=quality,
        profitability=profitability,
        growth=growth,
        leverage=leverage,
        valuation=valuation,
        financial_score=score,
        reasons=tuple(reasons),
    )
