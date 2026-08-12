from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from math import isfinite
from typing import Dict, Optional, Tuple

import pandas as pd

from .market_calendar import IDXMarketCalendar
from .market_return_provider import MarketReturnProvider
from .price_provider import PriceProvider


class EventStudyStatus(str, Enum):
    OBSERVED = "observed"
    INSUFFICIENT_ESTIMATION_DATA = "insufficient_estimation_data"
    MISSING_BENCHMARK_DATA = "missing_benchmark_data"
    MISSING_STOCK_DATA = "missing_stock_data"
    INSUFFICIENT_EVENT_DATA = "insufficient_event_data"
    PROVIDER_ERROR = "provider_error"
    INVALID_RETURN_DATA = "invalid_return_data"
    MODEL_FAILURE = "model_failure"
    INSUFFICIENT_CROSS_SECTION = "insufficient_cross_section"


@dataclass(frozen=True)
class AlignedReturn:
    offset: int
    trading_date: date
    stock_return: float
    market_return: float


@dataclass(frozen=True)
class HistoricalReturnData:
    ticker: str
    event_date: date
    returns: Tuple[AlignedReturn, ...]
    missing_stock_offsets: Tuple[int, ...]
    missing_benchmark_offsets: Tuple[int, ...]
    status: EventStudyStatus

    def by_offset(self) -> Dict[int, AlignedReturn]:
        return {item.offset: item for item in self.returns}


class HistoricalReturnDataProvider:
    """Builds Phase-2-calendar-aligned stock and IHSG simple daily returns."""

    def __init__(self, stock_provider: Optional[PriceProvider] = None,
                 market_provider: Optional[MarketReturnProvider] = None,
                 calendar: Optional[IDXMarketCalendar] = None) -> None:
        self.stock_provider = stock_provider or PriceProvider()
        self.market_provider = market_provider or MarketReturnProvider()
        self.calendar = calendar or IDXMarketCalendar()

    def get_returns(self, ticker: str, event_date: date,
                    start_offset: int = -250, end_offset: int = 10) -> HistoricalReturnData:
        dates = self._dates(event_date, start_offset - 1, end_offset)
        request_start, request_end = dates[start_offset - 1], dates[end_offset] + timedelta(days=1)
        stock, stock_status = self._history(self.stock_provider, ticker, request_start, request_end)
        market, market_status = self._history(self.market_provider, None, request_start, request_end)
        if stock_status == "provider_error" or market_status == "provider_error":
            return HistoricalReturnData(ticker, event_date, (), (), (), EventStudyStatus.PROVIDER_ERROR)
        stock_prices, stock_invalid = self._prices(stock, "Adj Close")
        market_prices, market_invalid = self._prices(market, "Close")
        records, missing_stock, missing_market, invalid = [], [], [], False
        for offset in range(start_offset, end_offset + 1):
            current, previous = dates[offset], dates[offset - 1]
            s0, s1 = stock_prices.get(previous), stock_prices.get(current)
            m0, m1 = market_prices.get(previous), market_prices.get(current)
            if current in stock_invalid or previous in stock_invalid or (s0 is not None and s0 <= 0) or (s1 is not None and s1 <= 0):
                invalid = True
                continue
            if current in market_invalid or previous in market_invalid or (m0 is not None and m0 <= 0) or (m1 is not None and m1 <= 0):
                invalid = True
                continue
            if s0 is None or s1 is None:
                missing_stock.append(offset)
                continue
            if m0 is None or m1 is None:
                missing_market.append(offset)
                continue
            records.append(AlignedReturn(offset, current, s1 / s0 - 1, m1 / m0 - 1))
        status = EventStudyStatus.INVALID_RETURN_DATA if invalid else EventStudyStatus.OBSERVED
        if status is EventStudyStatus.OBSERVED and missing_market:
            status = EventStudyStatus.MISSING_BENCHMARK_DATA
        if status is EventStudyStatus.OBSERVED and missing_stock:
            status = EventStudyStatus.MISSING_STOCK_DATA
        return HistoricalReturnData(ticker, event_date, tuple(records), tuple(missing_stock), tuple(missing_market), status)

    def _dates(self, event_date, start, end):
        dates = {0: event_date}
        current = event_date
        for offset in range(-1, start - 1, -1):
            current = self.calendar.previous_trading_day(current)
            dates[offset] = current
        current = event_date
        for offset in range(1, end + 1):
            current = self.calendar.next_trading_day(current)
            dates[offset] = current
        return dates

    @staticmethod
    def _history(provider, ticker, start, end):
        try:
            if hasattr(provider, "get_history_with_status"):
                return (provider.get_history_with_status(ticker, start.isoformat(), end.isoformat())
                        if ticker is not None else provider.get_history_with_status(start.isoformat(), end.isoformat()))
            return (provider.get_history(ticker, start.isoformat(), end.isoformat())
                    if ticker is not None else provider.get_history(start.isoformat(), end.isoformat())), "ok"
        except Exception:
            return pd.DataFrame(), "provider_error"

    @staticmethod
    def _prices(frame, field):
        if frame is None or frame.empty or "Date" not in frame or field not in frame:
            return {}, set()
        prices, invalid = {}, set()
        for _, row in frame.iterrows():
            parsed = pd.to_datetime(row["Date"], errors="coerce")
            if pd.isna(parsed):
                continue
            value = row[field]
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = None
            day = parsed.date()
            if value is None or not isfinite(value):
                invalid.add(day)
            else:
                prices[day] = value
        return prices, invalid
