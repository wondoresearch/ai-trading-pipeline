"""Composite financial provider with safe free-source fallback."""

from __future__ import annotations

from typing import Optional

from .models import FinancialObservation
from .idx_provider import IDXFinancialDataProvider
from .yahoo_provider import YahooFinanceFinancialProvider


class CompositeFinancialProvider:
    """Prefer IDX public data, then Yahoo Finance public data.

    `source_used` is explicit so downstream analysis can audit provenance.
    """

    name = "free_financial_composite"

    def __init__(
        self,
        cache_dir: str = "data/financial",
        idx_provider=None,
        yahoo_provider=None,
    ):
        self.idx = idx_provider or IDXFinancialDataProvider(
            f"{cache_dir}/idx"
        )
        self.yahoo = yahoo_provider or YahooFinanceFinancialProvider(
            f"{cache_dir}/yahoo"
        )
        self.last_source = None

    def fetch(self, ticker: str, quarterly: bool = False) -> Optional[FinancialObservation]:
        self.last_source = None

        try:
            obs = self.idx.latest_available(ticker)
        except Exception:
            obs = None

        if obs is not None:
            self.last_source = self.idx.name
            return obs

        try:
            obs = self.yahoo.fetch(ticker, quarterly=quarterly)
        except Exception:
            obs = None

        if obs is not None:
            self.last_source = self.yahoo.name
            return obs

        return None

    def status(self, ticker: str, quarterly: bool = False) -> dict:
        obs = self.fetch(ticker, quarterly=quarterly)
        if obs is None:
            return {
                "ticker": ticker.upper().replace(".JK", ""),
                "status": "not_found",
                "available": False,
                "provider": self.name,
                "source_used": None,
                "event_time_eligible": False,
                "observation": None,
            }

        event_time_eligible = bool(self.last_source == self.idx.name)
        return {
            "ticker": obs.ticker,
            "status": "available",
            "available": True,
            "provider": self.name,
            "source_used": self.last_source,
            "event_time_eligible": event_time_eligible,
            "observation": obs.to_dict(),
        }
