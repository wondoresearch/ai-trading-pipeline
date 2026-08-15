"""Market data provider abstraction."""
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, List

class MarketDataProvider(ABC):
    name = "unknown"

    @abstractmethod
    def history(self, ticker: str, start: str, end: str) -> List[Dict]:
        raise NotImplementedError

    def latest(self, ticker: str) -> Dict:
        rows = self.history(ticker, "1900-01-01", date.today().isoformat())
        if not rows:
            raise ValueError(f"No market data available for {ticker}")
        return max(rows, key=lambda row: row["date"])

    def health(self) -> Dict:
        return {"provider": self.name, "available": True}
