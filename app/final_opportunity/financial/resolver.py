"""Financial observation resolver with explicit availability semantics."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from .idx_provider import IDXFinancialDataProvider

@dataclass(frozen=True)
class FinancialResolution:
    ticker: str
    status: str
    observation: Optional[object]
    event_time_eligible: bool
    publication_date: Optional[str]
    warning: Optional[str]

    def to_dict(self):
        return {
            "ticker": self.ticker,
            "status": self.status,
            "event_time_eligible": self.event_time_eligible,
            "publication_date": self.publication_date,
            "warning": self.warning,
            "observation": self.observation.to_dict() if self.observation else None,
        }

class FinancialResolver:
    def __init__(self, provider=None):
        self.provider = provider or IDXFinancialDataProvider()

    def resolve(self, ticker: str, as_of: Optional[date] = None) -> FinancialResolution:
        ticker = ticker.upper().replace(".JK", "")
        obs = self.provider.latest_available(ticker)
        if obs is None:
            return FinancialResolution(
                ticker, "not_found", None, False, None,
                "No public IDX financial observation was found."
            )

        publication = getattr(obs, "fs_date", None) or None
        eligible = True
        if as_of and publication:
            try:
                eligible = datetime.fromisoformat(
                    publication[:10]
                ).date() <= as_of
            except ValueError:
                eligible = False

        return FinancialResolution(
            ticker=ticker,
            status="available",
            observation=obs,
            event_time_eligible=eligible,
            publication_date=publication,
            warning=None if eligible else "Financial observation is after analysis date.",
        )
