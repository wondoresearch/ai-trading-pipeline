"""Hard gate before market-dependent opportunity calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .provider_status import check_provider, provider_is_ready


class MarketReadinessError(RuntimeError):
    """Raised when market data cannot safely support downstream analysis."""


@dataclass(frozen=True)
class MarketReadiness:
    ready: bool
    status: str
    message: str
    provider_status: object

    def as_dict(self) -> dict:
        return {
            "ready": self.ready,
            "status": self.status,
            "message": self.message,
            "provider_status": self.provider_status.as_dict(),
        }


class MarketReadinessGate:
    def __init__(self, max_age_days: int = 3):
        self.max_age_days = max_age_days

    def evaluate(self, provider, tickers, today: date | None = None) -> MarketReadiness:
        status = check_provider(
            provider,
            tickers,
            max_age_days=self.max_age_days,
            today=today,
        )
        ready = provider_is_ready(status)
        return MarketReadiness(
            ready=ready,
            status=status.status,
            message=status.message,
            provider_status=status,
        )

    def require_ready(self, provider, tickers, today: date | None = None):
        result = self.evaluate(provider, tickers, today)
        if not result.ready:
            raise MarketReadinessError(
                f"Market data is not ready: {result.status}. {result.message}"
            )
        return result
