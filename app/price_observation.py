from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from .event_time import EventTimeResolution
from .market_calendar import IDXMarketCalendar
from .price_provider import PriceProvider


class PriceObservationStatus(str, Enum):
    OBSERVED = "observed"
    MISSING_BASELINE = "missing_baseline"
    MISSING_OBSERVATION = "missing_observation"
    INSUFFICIENT_FORWARD_DATA = "insufficient_forward_data"
    INVALID_DATA = "invalid_data"
    PROVIDER_ERROR = "provider_error"


class ObservationHorizon(str, Enum):
    EVENT_DAY = "event_day"
    T1 = "t1"
    T3 = "t3"
    T5 = "t5"
    T10 = "t10"

    @property
    def trading_day_offset(self) -> int:
        return {
            ObservationHorizon.EVENT_DAY: 0,
            ObservationHorizon.T1: 1,
            ObservationHorizon.T3: 3,
            ObservationHorizon.T5: 5,
            ObservationHorizon.T10: 10,
        }[self]


@dataclass(frozen=True)
class PriceObservation:
    horizon: ObservationHorizon
    observation_date: date
    observation_time: datetime
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    adjusted_close: Optional[float]
    volume: Optional[float]
    status: PriceObservationStatus

    @property
    def price(self) -> Optional[float]:
        """Canonical daily price used by downstream return calculations."""
        return self.adjusted_close

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon.value,
            "observation_date": self.observation_date.isoformat(),
            "observation_time": self.observation_time.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "adjusted_close": self.adjusted_close,
            "price": self.price,
            "volume": self.volume,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class BaselinePrice:
    baseline_date: date
    baseline_time: datetime
    raw_close: Optional[float]
    adjusted_close: Optional[float]
    status: PriceObservationStatus

    @property
    def price(self) -> Optional[float]:
        return self.adjusted_close

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_date": self.baseline_date.isoformat(),
            "baseline_time": self.baseline_time.isoformat(),
            "raw_close": self.raw_close,
            "adjusted_close": self.adjusted_close,
            "price": self.price,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class PriceObservationSet:
    event_id: str
    ticker: str
    effective_time: datetime
    baseline: BaselinePrice
    observations: Tuple[PriceObservation, ...]
    status: PriceObservationStatus
    price_source: str
    price_granularity: str

    def __post_init__(self) -> None:
        if self.effective_time.tzinfo is None or self.effective_time.utcoffset() is None:
            raise ValueError("effective_time must be timezone-aware")
        if self.price_granularity != "daily":
            raise ValueError("price_granularity must be daily")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "effective_time": self.effective_time.isoformat(),
            "baseline": self.baseline.to_dict(),
            "observations": [item.to_dict() for item in self.observations],
            "status": self.status.value,
            "price_source": self.price_source,
            "price_granularity": self.price_granularity,
        }


class HistoricalPriceObservationEngine:
    """Resolve daily historical OHLCV observations from Phase 3 event time."""

    TIMEZONE = "Asia/Jakarta"
    PRICE_SOURCE = "yahoo"
    PRICE_GRANULARITY = "daily"
    HORIZONS = (
        ObservationHorizon.EVENT_DAY,
        ObservationHorizon.T1,
        ObservationHorizon.T3,
        ObservationHorizon.T5,
        ObservationHorizon.T10,
    )

    def __init__(
        self,
        price_provider: Optional[PriceProvider] = None,
        calendar: Optional[IDXMarketCalendar] = None,
    ) -> None:
        self.price_provider = price_provider or PriceProvider()
        self.calendar = calendar or IDXMarketCalendar()
        self.tz = ZoneInfo(self.TIMEZONE)

    def resolve(self, resolution: EventTimeResolution) -> PriceObservationSet:
        if resolution.effective_time.tzinfo is None or resolution.effective_time.utcoffset() is None:
            raise ValueError("effective_time must be timezone-aware")

        effective_local = resolution.effective_time.astimezone(self.tz)
        event_date = effective_local.date()
        if not self.calendar.is_trading_day(event_date):
            raise ValueError("effective_time must resolve to a trading day")

        trading_dates = self._required_trading_dates(event_date)
        baseline_date = self.calendar.previous_trading_day(event_date)

        start_date = baseline_date
        end_date = trading_dates[-1] + timedelta(days=1)

        try:
            if hasattr(self.price_provider, "get_history_with_status"):
                history, provider_status = self.price_provider.get_history_with_status(
                    resolution.ticker,
                    start_date.isoformat(),
                    end_date.isoformat(),
                )
                if provider_status == "provider_error":
                    return self._provider_error_result(resolution, baseline_date, trading_dates)
            else:
                history = self.price_provider.get_history(
                    resolution.ticker,
                    start_date.isoformat(),
                    end_date.isoformat(),
                )
        except Exception:
            return self._provider_error_result(resolution, baseline_date, trading_dates)

        try:
            normalized, invalid_dates = self._normalize_history(history)
        except ValueError:
            return self._invalid_data_result(resolution, baseline_date, trading_dates)

        baseline = self._build_baseline(baseline_date, normalized, invalid_dates)
        observations = [
            self._build_observation(
                horizon,
                trading_dates[horizon.trading_day_offset],
                normalized,
                invalid_dates,
            )
            for horizon in self.HORIZONS
        ]

        status = self._overall_status(baseline, observations)

        return PriceObservationSet(
            event_id=resolution.event_id,
            ticker=resolution.ticker,
            effective_time=effective_local,
            baseline=baseline,
            observations=tuple(observations),
            status=status,
            price_source=self.PRICE_SOURCE,
            price_granularity=self.PRICE_GRANULARITY,
        )

    def _required_trading_dates(self, event_date: date) -> List[date]:
        dates = [event_date]
        current = event_date
        for _ in range(10):
            current = self.calendar.next_trading_day(current)
            dates.append(current)
        return dates

    @staticmethod
    def _normalize_history(df: pd.DataFrame):
        if df is None or df.empty:
            return {}, set()

        frame = df.copy()
        if "Date" not in frame.columns:
            if "Datetime" in frame.columns:
                frame = frame.rename(columns={"Datetime": "Date"})
            else:
                raise ValueError("Price data does not contain Date/Datetime column")

        parsed = pd.to_datetime(frame["Date"], errors="coerce")
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
        else:
            parsed = parsed.dt.tz_localize(None)

        frame["Date"] = parsed.dt.date
        frame = frame.dropna(subset=["Date"])
        frame = frame.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

        required = {"Close", "Adj Close"}
        if not required.issubset(frame.columns):
            raise ValueError("Price data must contain Close and Adj Close columns")

        records: Dict[date, Dict[str, Any]] = {}
        invalid_dates = set()
        numeric_columns = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
        for _, row in frame.iterrows():
            record = {
                column: HistoricalPriceObservationEngine._to_float(row.get(column))
                for column in numeric_columns
            }
            records[row["Date"]] = record
            if record["Close"] is None or record["Adj Close"] is None:
                invalid_dates.add(row["Date"])

        return records, invalid_dates

    @staticmethod
    def _to_float(value) -> Optional[float]:
        if value is None or pd.isna(value):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if pd.notna(result) else None

    def _build_baseline(
        self,
        baseline_date: date,
        records: Dict[date, Dict[str, Any]],
        invalid_dates: Iterable[date],
    ) -> BaselinePrice:
        if baseline_date not in records:
            return BaselinePrice(
                baseline_date=baseline_date,
                baseline_time=self._close_time(baseline_date),
                raw_close=None,
                adjusted_close=None,
                status=PriceObservationStatus.MISSING_BASELINE,
            )

        record = records[baseline_date]
        if baseline_date in invalid_dates:
            status = PriceObservationStatus.INVALID_DATA
        else:
            status = PriceObservationStatus.OBSERVED

        return BaselinePrice(
            baseline_date=baseline_date,
            baseline_time=self._close_time(baseline_date),
            raw_close=record["Close"],
            adjusted_close=record["Adj Close"],
            status=status,
        )

    def _build_observation(
        self,
        horizon: ObservationHorizon,
        observation_date: date,
        records: Dict[date, Dict[str, Any]],
        invalid_dates: Iterable[date],
    ) -> PriceObservation:
        record = records.get(observation_date)
        if record is None:
            return PriceObservation(
                horizon=horizon,
                observation_date=observation_date,
                observation_time=self._close_time(observation_date),
                open=None,
                high=None,
                low=None,
                close=None,
                adjusted_close=None,
                volume=None,
                status=PriceObservationStatus.MISSING_OBSERVATION,
            )

        status = (
            PriceObservationStatus.INVALID_DATA
            if observation_date in invalid_dates
            else PriceObservationStatus.OBSERVED
        )
        return PriceObservation(
            horizon=horizon,
            observation_date=observation_date,
            observation_time=self._close_time(observation_date),
            open=record["Open"],
            high=record["High"],
            low=record["Low"],
            close=record["Close"],
            adjusted_close=record["Adj Close"],
            volume=record["Volume"],
            status=status,
        )

    def _close_time(self, session_date: date) -> datetime:
        close = self.calendar.regular_close(session_date)
        if close is None:
            raise ValueError(f"No IDX regular close for {session_date}")
        return close.astimezone(self.tz)

    @staticmethod
    def _overall_status(
        baseline: BaselinePrice,
        observations: List[PriceObservation],
    ) -> PriceObservationStatus:
        if baseline.status == PriceObservationStatus.MISSING_BASELINE:
            return PriceObservationStatus.MISSING_BASELINE
        if baseline.status == PriceObservationStatus.INVALID_DATA:
            return PriceObservationStatus.INVALID_DATA
        if any(item.status == PriceObservationStatus.INVALID_DATA for item in observations):
            return PriceObservationStatus.INVALID_DATA
        if any(item.status == PriceObservationStatus.MISSING_OBSERVATION for item in observations):
            return PriceObservationStatus.INSUFFICIENT_FORWARD_DATA
        return PriceObservationStatus.OBSERVED


    def _invalid_data_result(
        self,
        resolution: EventTimeResolution,
        baseline_date: date,
        trading_dates: List[date],
    ) -> PriceObservationSet:
        baseline = BaselinePrice(
            baseline_date=baseline_date,
            baseline_time=self._close_time(baseline_date),
            raw_close=None,
            adjusted_close=None,
            status=PriceObservationStatus.INVALID_DATA,
        )
        observations = [
            PriceObservation(
                horizon=horizon,
                observation_date=trading_dates[horizon.trading_day_offset],
                observation_time=self._close_time(trading_dates[horizon.trading_day_offset]),
                open=None,
                high=None,
                low=None,
                close=None,
                adjusted_close=None,
                volume=None,
                status=PriceObservationStatus.INVALID_DATA,
            )
            for horizon in self.HORIZONS
        ]
        return PriceObservationSet(
            event_id=resolution.event_id,
            ticker=resolution.ticker,
            effective_time=resolution.effective_time.astimezone(self.tz),
            baseline=baseline,
            observations=tuple(observations),
            status=PriceObservationStatus.INVALID_DATA,
            price_source=self.PRICE_SOURCE,
            price_granularity=self.PRICE_GRANULARITY,
        )

    def _provider_error_result(
        self,
        resolution: EventTimeResolution,
        baseline_date: date,
        trading_dates: List[date],
    ) -> PriceObservationSet:
        baseline = BaselinePrice(
            baseline_date=baseline_date,
            baseline_time=self._close_time(baseline_date),
            raw_close=None,
            adjusted_close=None,
            status=PriceObservationStatus.PROVIDER_ERROR,
        )
        observations = [
            PriceObservation(
                horizon=horizon,
                observation_date=trading_dates[horizon.trading_day_offset],
                observation_time=self._close_time(trading_dates[horizon.trading_day_offset]),
                open=None,
                high=None,
                low=None,
                close=None,
                adjusted_close=None,
                volume=None,
                status=PriceObservationStatus.PROVIDER_ERROR,
            )
            for horizon in self.HORIZONS
        ]
        return PriceObservationSet(
            event_id=resolution.event_id,
            ticker=resolution.ticker,
            effective_time=resolution.effective_time.astimezone(self.tz),
            baseline=baseline,
            observations=tuple(observations),
            status=PriceObservationStatus.PROVIDER_ERROR,
            price_source=self.PRICE_SOURCE,
            price_granularity=self.PRICE_GRANULARITY,
        )
