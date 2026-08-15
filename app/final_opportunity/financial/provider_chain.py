"""Financial provider chain with explicit provenance and temporal safety."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FinancialProviderResult:
    observation: object | None
    provider: str
    status: str
    event_time_eligible: bool
    publication_date: Optional[str]
    warning: Optional[str] = None

    def to_dict(self):
        return {
            "provider": self.provider,
            "status": self.status,
            "event_time_eligible": self.event_time_eligible,
            "publication_date": self.publication_date,
            "warning": self.warning,
            "available": self.observation is not None,
            "observation": (
                self.observation.to_dict()
                if self.observation is not None else None
            ),
        }


class FinancialProviderChain:
    """Prefer official IDX; fall back to free Yahoo only for current context."""

    def __init__(self, idx_provider, yahoo_provider=None):
        self.idx = idx_provider
        self.yahoo = yahoo_provider

    def fetch(self, ticker: str) -> FinancialProviderResult:
        try:
            obs = self.idx.latest_available(ticker)
            if obs is not None:
                return FinancialProviderResult(
                    observation=obs,
                    provider=self.idx.name,
                    status="available",
                    event_time_eligible=False,
                    publication_date=None,
                )
        except Exception:
            pass

        if self.yahoo is not None:
            try:
                obs = self.yahoo.fetch(ticker)
                if obs is not None:
                    return FinancialProviderResult(
                        observation=obs,
                        provider=self.yahoo.name,
                        status="fallback_available",
                        event_time_eligible=False,
                        publication_date=None,
                        warning=(
                            "Fallback source has no verified publication date; "
                            "excluded from historical event-time attribution."
                        ),
                    )
            except Exception as exc:
                return FinancialProviderResult(
                    observation=None,
                    provider=self.yahoo.name,
                    status="error",
                    event_time_eligible=False,
                    publication_date=None,
                    warning=str(exc),
                )

        return FinancialProviderResult(
            observation=None,
            provider=self.idx.name,
            status="not_found",
            event_time_eligible=False,
            publication_date=None,
        )
