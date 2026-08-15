"""Free/local IDX EOD market-data provider.

The provider deliberately avoids Yahoo Finance and paid APIs. It reads normalized
CSV files produced by the market importer from data/market_eod/.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from .market_provider import MarketDataProvider


def _parse_date(value: str) -> date:
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unsupported date: {value}")


def _number(value: str) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip().replace(",", "")
    if value in ("", "-", "--", "null", "None"):
        return None
    return float(value)


@dataclass(frozen=True)
class MarketBar:
    date: date
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[float]


class FreeIDXMarketData(MarketDataProvider):
    """Read normalized IDX EOD CSV files from disk."""

    name = "local_idx"

    def __init__(self, data_dir: str = "data/market_eod"):
        self.data_dir = Path(data_dir)

    @staticmethod
    def provider_symbol(ticker: str) -> str:
        return ticker.upper().replace(".JK", "")

    def _path(self, ticker: str) -> Path:
        return self.data_dir / f"{self.provider_symbol(ticker)}.csv"

    def history(self, ticker: str, start: str, end: str) -> List[Dict]:
        path = self._path(ticker)
        if not path.exists():
            raise ValueError(
                f"No local IDX EOD data for {ticker}. "
                f"Import an official IDX EOD CSV/ZIP first."
            )

        start_d = _parse_date(start)
        end_d = _parse_date(end)
        rows: List[Dict] = []

        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                d = _parse_date(raw["date"])
                if start_d <= d <= end_d:
                    rows.append({
                        "date": d.isoformat(),
                        "open": _number(raw.get("open")),
                        "high": _number(raw.get("high")),
                        "low": _number(raw.get("low")),
                        "close": _number(raw.get("close")),
                        "volume": _number(raw.get("volume")),
                    })

        rows.sort(key=lambda x: x["date"])
        return rows


# Backward-compatible name for old imports. It no longer calls Yahoo.
YahooFinanceMarketData = FreeIDXMarketData
