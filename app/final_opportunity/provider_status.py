"""Provider status and freshness evaluation.

Freshness is deterministic: callers provide the reference date. A provider is
stale when ANY requested ticker is older than max_age_days. Missing tickers are
degraded. No data is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    status: str
    available: bool
    fresh: bool
    tickers: List[str]
    stale_tickers: List[str]
    missing_tickers: List[str]
    rows: int
    oldest_date: Optional[str]
    latest_date: Optional[str]
    age_days: Optional[int]
    max_age_days: int
    message: str

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "status": self.status,
            "available": self.available,
            "fresh": self.fresh,
            "tickers": self.tickers,
            "stale_tickers": self.stale_tickers,
            "missing_tickers": self.missing_tickers,
            "rows": self.rows,
            "oldest_date": self.oldest_date,
            "latest_date": self.latest_date,
            "age_days": self.age_days,
            "max_age_days": self.max_age_days,
            "message": self.message,
        }


def check_provider(provider, tickers, max_age_days: int = 3,
                   today: Optional[date] = None) -> ProviderStatus:
    requested = [str(t).upper().replace(".JK", "") for t in tickers]
    reference = today or date.today()

    if not requested:
        return ProviderStatus(
            provider=provider.name, status="unavailable", available=False,
            fresh=False, tickers=[], stale_tickers=[], missing_tickers=[],
            rows=0, oldest_date=None, latest_date=None, age_days=None,
            max_age_days=max_age_days, message="No tickers requested.",
        )

    latest_by_ticker = {}
    rows_total = 0

    for ticker in requested:
        try:
            rows = provider.history(ticker, "1900-01-01", reference.isoformat())
        except (FileNotFoundError, ValueError, KeyError):
            rows = []
        rows_total += len(rows)
        if rows:
            latest = max(str(row["date"])[:10] for row in rows)
            latest_by_ticker[ticker] = date.fromisoformat(latest)

    missing = [t for t in requested if t not in latest_by_ticker]
    stale = [
        t for t, d in latest_by_ticker.items()
        if (reference - d).days > max_age_days
    ]

    all_dates = list(latest_by_ticker.values())
    latest_date = max(all_dates).isoformat() if all_dates else None
    oldest_date = min(all_dates).isoformat() if all_dates else None
    age_days = max((reference - d).days for d in all_dates) if all_dates else None

    if missing and len(missing) == len(requested):
        status = "unavailable"
        message = "No market data available for requested tickers."
    elif missing:
        status = "degraded"
        message = "One or more requested tickers have no market data."
    elif stale:
        status = "stale"
        message = "One or more requested tickers exceed the freshness threshold."
    else:
        status = "healthy"
        message = "All requested tickers are available and fresh."

    return ProviderStatus(
        provider=provider.name,
        status=status,
        available=status != "unavailable",
        fresh=status == "healthy",
        tickers=requested,
        stale_tickers=stale,
        missing_tickers=missing,
        rows=rows_total,
        oldest_date=oldest_date,
        latest_date=latest_date,
        age_days=age_days,
        max_age_days=max_age_days,
        message=message,
    )


def provider_is_ready(status: ProviderStatus) -> bool:
    return status.status == "healthy"
