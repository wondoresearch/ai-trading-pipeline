from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from math import isfinite
from typing import Any, Dict, Optional, Tuple

from .price_observation import (
    ObservationHorizon,
    PriceObservation,
    PriceObservationSet,
    PriceObservationStatus,
)


class ReturnStatus(str, Enum):
    """Explicit availability state for a Phase 5 return value."""

    OBSERVED = "observed"
    MISSING_BASELINE = "missing_baseline"
    MISSING_EVENT_DAY = "missing_event_day"
    MISSING_FORWARD_OBSERVATION = "missing_forward_observation"
    PROVIDER_ERROR = "provider_error"
    INVALID_DATA = "invalid_data"
    ZERO_BASELINE = "zero_baseline"
    ZERO_EVENT_DAY = "zero_event_day"


@dataclass(frozen=True)
class EventDayReturn:
    """The Phase 5 event-day observation and baseline-relative return."""

    observation_date: date
    observation_time: datetime
    adjusted_close: Optional[float]
    event_day_return: Optional[float]
    status: ReturnStatus

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_date": self.observation_date.isoformat(),
            "observation_time": self.observation_time.isoformat(),
            "adjusted_close": self.adjusted_close,
            "event_day_return": self.event_day_return,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class ForwardReturnObservation:
    """One forward horizon with baseline-relative and event-day-relative returns."""

    horizon: ObservationHorizon
    observation_date: date
    observation_time: datetime
    adjusted_close: Optional[float]
    cumulative_return: Optional[float]
    cumulative_status: ReturnStatus
    forward_return: Optional[float]
    forward_status: ReturnStatus

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon.value,
            "observation_date": self.observation_date.isoformat(),
            "observation_time": self.observation_time.isoformat(),
            "adjusted_close": self.adjusted_close,
            "cumulative_return": self.cumulative_return,
            "cumulative_status": self.cumulative_status.value,
            "forward_return": self.forward_return,
            "forward_status": self.forward_status.value,
        }


@dataclass(frozen=True)
class ReturnResult:
    """Canonical, JSON-serializable Phase 5 result for one ticker event."""

    event_id: str
    ticker: str
    effective_time: datetime
    baseline_date: date
    baseline_price: Optional[float]
    baseline_status: ReturnStatus
    event_day: EventDayReturn
    forward_observations: Tuple[ForwardReturnObservation, ...]
    status: ReturnStatus
    price_source: str
    price_granularity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "effective_time": self.effective_time.isoformat(),
            "baseline_date": self.baseline_date.isoformat(),
            "baseline_price": self.baseline_price,
            "baseline_status": self.baseline_status.value,
            "event_day": self.event_day.to_dict(),
            "forward_observations": [item.to_dict() for item in self.forward_observations],
            "status": self.status.value,
            "price_source": self.price_source,
            "price_granularity": self.price_granularity,
        }


class ReturnEngine:
    """Calculate Phase 5 returns only from Phase 4 adjusted-close prices."""

    FORWARD_HORIZONS = (
        ObservationHorizon.T1,
        ObservationHorizon.T3,
        ObservationHorizon.T5,
        ObservationHorizon.T10,
    )

    def calculate(self, observations: PriceObservationSet) -> ReturnResult:
        """Return Phase 5 values without mutating the Phase 4 input."""
        by_horizon = {item.horizon: item for item in observations.observations}
        baseline_price = observations.baseline.price
        baseline_status = self._baseline_status(
            observations.baseline.status, baseline_price
        )
        event_day = self._event_day_return(
            baseline_price,
            baseline_status,
            by_horizon.get(ObservationHorizon.EVENT_DAY),
        )
        forwards = tuple(
            self._forward_return(
                horizon,
                by_horizon.get(horizon),
                baseline_price,
                baseline_status,
                event_day,
            )
            for horizon in self.FORWARD_HORIZONS
        )
        return ReturnResult(
            event_id=observations.event_id,
            ticker=observations.ticker,
            effective_time=observations.effective_time,
            baseline_date=observations.baseline.baseline_date,
            baseline_price=baseline_price,
            baseline_status=baseline_status,
            event_day=event_day,
            forward_observations=forwards,
            status=self._overall_status(baseline_status, event_day, forwards),
            price_source=observations.price_source,
            price_granularity=observations.price_granularity,
        )

    @staticmethod
    def _baseline_status(status, price):
        mapped = ReturnEngine._price_status(status, False)
        if mapped is not ReturnStatus.OBSERVED:
            return mapped
        if not ReturnEngine._is_finite(price):
            return ReturnStatus.INVALID_DATA
        if price == 0:
            return ReturnStatus.ZERO_BASELINE
        return ReturnStatus.OBSERVED

    def _event_day_return(self, baseline_price, baseline_status, observation):
        if observation is None:
            return EventDayReturn(
                date.min, datetime.min, None, None, ReturnStatus.MISSING_EVENT_DAY
            )
        event_status = self._price_status(observation.status, True)
        event_price = observation.price
        if event_status is ReturnStatus.OBSERVED and not self._is_finite(event_price):
            event_status = ReturnStatus.INVALID_DATA
        if event_status is ReturnStatus.OBSERVED and event_price == 0:
            event_status = ReturnStatus.ZERO_EVENT_DAY
        status = self._return_status(baseline_status, event_status)
        return EventDayReturn(
            observation.observation_date,
            observation.observation_time,
            event_price,
            self._simple_return(baseline_price, event_price)
            if status is ReturnStatus.OBSERVED
            else None,
            status,
        )

    def _forward_return(
        self, horizon, observation, baseline_price, baseline_status, event_day
    ):
        if observation is None:
            return ForwardReturnObservation(
                horizon, date.min, datetime.min, None, None,
                ReturnStatus.MISSING_FORWARD_OBSERVATION, None,
                ReturnStatus.MISSING_FORWARD_OBSERVATION,
            )
        observation_status = self._price_status(observation.status, False)
        observation_price = observation.price
        if observation_status is ReturnStatus.OBSERVED and not self._is_finite(
            observation_price
        ):
            observation_status = ReturnStatus.INVALID_DATA
        cumulative_status = self._return_status(baseline_status, observation_status)
        event_day_price_status = self._event_day_price_status(event_day)
        forward_status = self._return_status(
            event_day_price_status, observation_status
        )
        return ForwardReturnObservation(
            horizon,
            observation.observation_date,
            observation.observation_time,
            observation_price,
            self._simple_return(baseline_price, observation_price)
            if cumulative_status is ReturnStatus.OBSERVED
            else None,
            cumulative_status,
            self._simple_return(event_day.adjusted_close, observation_price)
            if forward_status is ReturnStatus.OBSERVED
            else None,
            forward_status,
        )

    @staticmethod
    def _price_status(status, is_event_day):
        if status is PriceObservationStatus.OBSERVED:
            return ReturnStatus.OBSERVED
        if status is PriceObservationStatus.MISSING_BASELINE:
            return ReturnStatus.MISSING_BASELINE
        if status is PriceObservationStatus.MISSING_OBSERVATION:
            return (
                ReturnStatus.MISSING_EVENT_DAY
                if is_event_day
                else ReturnStatus.MISSING_FORWARD_OBSERVATION
            )
        if status is PriceObservationStatus.INSUFFICIENT_FORWARD_DATA:
            return ReturnStatus.MISSING_FORWARD_OBSERVATION
        if status is PriceObservationStatus.PROVIDER_ERROR:
            return ReturnStatus.PROVIDER_ERROR
        return ReturnStatus.INVALID_DATA

    @staticmethod
    def _return_status(denominator_status, numerator_status):
        for status in (
            ReturnStatus.PROVIDER_ERROR,
            ReturnStatus.INVALID_DATA,
            ReturnStatus.ZERO_BASELINE,
            ReturnStatus.ZERO_EVENT_DAY,
            ReturnStatus.MISSING_BASELINE,
            ReturnStatus.MISSING_EVENT_DAY,
            ReturnStatus.MISSING_FORWARD_OBSERVATION,
        ):
            if denominator_status is status or numerator_status is status:
                return status
        return ReturnStatus.OBSERVED

    @staticmethod
    def _event_day_price_status(event_day):
        """Use the event-day price quality, not its baseline-relative status."""
        if not ReturnEngine._is_finite(event_day.adjusted_close):
            return event_day.status
        if event_day.adjusted_close == 0:
            return ReturnStatus.ZERO_EVENT_DAY
        return ReturnStatus.OBSERVED

    @staticmethod
    def _simple_return(denominator, numerator):
        assert denominator is not None
        assert numerator is not None
        return numerator / denominator - 1

    @staticmethod
    def _is_finite(price):
        return price is not None and isfinite(price)

    @staticmethod
    def _overall_status(baseline_status, event_day, forwards):
        statuses = [baseline_status, event_day.status]
        for item in forwards:
            statuses.extend((item.cumulative_status, item.forward_status))
        for status in (
            ReturnStatus.PROVIDER_ERROR,
            ReturnStatus.INVALID_DATA,
            ReturnStatus.ZERO_BASELINE,
            ReturnStatus.ZERO_EVENT_DAY,
            ReturnStatus.MISSING_BASELINE,
            ReturnStatus.MISSING_EVENT_DAY,
            ReturnStatus.MISSING_FORWARD_OBSERVATION,
        ):
            if status in statuses:
                return status
        return ReturnStatus.OBSERVED
