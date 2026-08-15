from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict


@dataclass(frozen=True)
class FinancialObservation:
    ticker: str
    sector: str
    sub_industry: str
    sub_industry_code: str
    fs_date: str
    fiscal_year_end: str
    statement_type: str
    auditor_opinion: str
    assets: Optional[float]
    liabilities: Optional[float]
    equity: Optional[float]
    sales: Optional[float]
    ebt: Optional[float]
    profit: Optional[float]
    profit_attributed: Optional[float]
    eps: Optional[float]
    book_value: Optional[float]
    pe: Optional[float]
    pbv: Optional[float]
    debt_equity: Optional[float]
    roa: Optional[float]
    roe: Optional[float]
    npm: Optional[float]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True)
class FinancialFeatureSet:
    ticker: str
    sector_group: str
    observation_date: str
    quality: float
    profitability: Optional[float]
    growth: Optional[float]
    leverage: Optional[float]
    valuation: Optional[float]
    financial_score: float
    reasons: tuple[str, ...]

    sub_industry: str | None = None
    profile: str | None = None
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d
