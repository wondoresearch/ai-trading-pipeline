"""User-defined stock universe.

This module deliberately has no hard-coded stock list.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

_TICKER_PATTERN = re.compile(r"^[A-Z0-9.-]+$")


@dataclass(frozen=True)
class StockUniverse:
    tickers: tuple[str, ...]

    @classmethod
    def from_tickers(cls, tickers: Iterable[str]) -> "StockUniverse":
        values = []
        for ticker in tickers:
            if not isinstance(ticker, str):
                raise ValueError("ticker must be a string")
            value = ticker.strip().upper()
            if not value:
                raise ValueError("ticker must not be empty")
            if not _TICKER_PATTERN.fullmatch(value):
                raise ValueError(f"invalid ticker format: {ticker}")
            values.append(value)

        if not values:
            raise ValueError("stock universe must not be empty")
        if len(set(values)) != len(values):
            raise ValueError("stock universe contains duplicate tickers")

        return cls(tuple(values))

    def __contains__(self, ticker: str) -> bool:
        return ticker.strip().upper() in self.tickers

    def __len__(self) -> int:
        return len(self.tickers)
