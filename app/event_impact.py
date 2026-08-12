"""Phase 7 canonical event-impact labeling.

This layer consumes canonical outputs from Phases 1-6. It does not fetch
market data or recalculate event time, returns, abnormal returns, or CAR.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from math import isfinite
from typing import Any, Mapping, Optional


class ImpactDirection(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class ImpactStrength(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class StatisticalSignificance(str, Enum):
    SIGNIFICANT = "SIGNIFICANT"
    INSIGNIFICANT = "INSIGNIFICANT"
    UNKNOWN = "UNKNOWN"


class SentimentAlignment(str, Enum):
    ALIGNED = "ALIGNED"
    CONTRADICTED = "CONTRADICTED"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class ImpactLabel(str, Enum):
    POSITIVE_SIGNIFICANT = "POSITIVE_SIGNIFICANT"
    POSITIVE_INSIGNIFICANT = "POSITIVE_INSIGNIFICANT"
    NEGATIVE_SIGNIFICANT = "NEGATIVE_SIGNIFICANT"
    NEGATIVE_INSIGNIFICANT = "NEGATIVE_INSIGNIFICANT"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class ImpactStatus(str, Enum):
    OBSERVED = "OBSERVED"
    MISSING_SENTIMENT = "MISSING_SENTIMENT"
    MISSING_RETURN = "MISSING_RETURN"
    MISSING_EVENT_STUDY = "MISSING_EVENT_STUDY"
    INSUFFICIENT_STATISTICS = "INSUFFICIENT_STATISTICS"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class ImpactThresholds:
    medium_abs_car: float = 0.02
    high_abs_car: float = 0.05

    def __post_init__(self) -> None:
        if (
            not isinstance(self.medium_abs_car, (int, float))
            or isinstance(self.medium_abs_car, bool)
            or not isinstance(self.high_abs_car, (int, float))
            or isinstance(self.high_abs_car, bool)
            or not isfinite(float(self.medium_abs_car))
            or not isfinite(float(self.high_abs_car))
            or self.medium_abs_car < 0
            or self.high_abs_car < self.medium_abs_car
        ):
            raise ValueError("invalid impact thresholds")


@dataclass(frozen=True)
class EventImpactRecord:
    event_id: Optional[str]
    ticker: Optional[str]
    sentiment: Optional[str]
    effective_time: Optional[str]
    realized_return: Optional[float]
    abnormal_return: Optional[float]
    car: Optional[float]
    statistic: Optional[float]
    p_value: Optional[float]
    significant: Optional[bool]
    impact_direction: ImpactDirection
    impact_strength: ImpactStrength
    statistical_significance: StatisticalSignificance
    sentiment_alignment: SentimentAlignment
    impact_label: ImpactLabel
    status: ImpactStatus

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return {
            key: value.value if isinstance(value, Enum) else value
            for key, value in result.items()
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _get(source: Any, *names: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return None
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    to_dict = getattr(source, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, Mapping):
            for name in names:
                if name in data:
                    return data[name]
    return None


def _float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _enum_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _event_id(event: Any) -> Optional[str]:
    value = _get(event, "event_id", "id")
    return None if value is None else str(value)


def _ticker(event: Any, return_result: Any) -> Optional[str]:
    value = _get(event, "ticker", "symbol")
    if value is None:
        value = _get(return_result, "ticker", "symbol")
    return None if value is None else str(value)


def _effective_time(event_time: Any) -> Optional[str]:
    value = _get(event_time, "effective_time")
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _sentiment(event: Any, sentiment: Any) -> Optional[str]:
    value = _get(sentiment, "label", "sentiment", "polarity", "class")
    if value is None:
        value = _get(event, "sentiment", "sentiment_label")
    return _enum_text(value)


def _phase6_car(event_study: Any, window_name: str) -> Optional[float]:
    """Read exactly one explicitly configured Phase-6 CAR window."""
    if event_study is None:
        return None

    direct_window = _get(event_study, window_name)
    if direct_window is not None:
        return _float(_get(direct_window, "car"))

    windows = _get(event_study, "windows")
    if windows is None:
        return None

    for window in windows:
        if _get(window, "window", "name", "label") == window_name:
            return _float(_get(window, "car"))
    return None


def _phase6_abnormal_return(event_study: Any) -> Optional[float]:
    return _float(_get(event_study, "abnormal_return", "ar", "event_day_abnormal_return"))


def _phase6_statistic(event_study: Any) -> Optional[float]:
    return _float(_get(event_study, "statistic", "test_statistic", "t_statistic"))


def _phase6_p_value(event_study: Any) -> Optional[float]:
    return _float(_get(event_study, "p_value", "pvalue"))


def _phase6_significant(event_study: Any) -> Optional[bool]:
    value = _get(event_study, "significant")
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("Phase 6 significant must be bool or None")
    return value


def _realized_return(return_result: Any) -> Optional[float]:
    return _float(_get(return_result, "event_day_return", "realized_return"))


class EventImpactEngine:
    """Build one canonical event/ticker impact record."""

    def __init__(
        self,
        thresholds: Optional[ImpactThresholds] = None,
        alpha: float = 0.05,
        car_window: str = "car_0_1",
    ) -> None:
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise ValueError("alpha must be numeric")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if not car_window:
            raise ValueError("car_window must not be empty")
        self.thresholds = thresholds or ImpactThresholds()
        self.alpha = float(alpha)
        self.car_window = car_window

    def build(
        self,
        event: Any,
        sentiment: Any,
        event_time: Any,
        return_result: Any,
        event_study: Any,
    ) -> EventImpactRecord:
        if event is None:
            return self._invalid(None, None, None, None, None)
        if event_time is None:
            return self._invalid(_event_id(event), _ticker(event, return_result), _sentiment(event, sentiment), None, _realized_return(return_result))

        event_id = _event_id(event)
        ticker = _ticker(event, return_result)
        sentiment_value = _sentiment(event, sentiment)
        effective_time = _effective_time(event_time)
        realized = _realized_return(return_result)

        try:
            if sentiment_value is None:
                return self._missing_sentiment_record(event_id, ticker, effective_time, realized)
            if return_result is None or realized is None:
                return self._record(event_id, ticker, sentiment_value, effective_time, realized, event_study, ImpactStatus.MISSING_RETURN)
            if event_study is None:
                return self._record(event_id, ticker, sentiment_value, effective_time, realized, None, ImpactStatus.MISSING_EVENT_STUDY)

            car = _phase6_car(event_study, self.car_window)
            abnormal_return = _phase6_abnormal_return(event_study)
            statistic = _phase6_statistic(event_study)
            p_value = _phase6_p_value(event_study)
            significant = _phase6_significant(event_study)

            if car is None:
                return self._record(event_id, ticker, sentiment_value, effective_time, realized, event_study, ImpactStatus.INSUFFICIENT_STATISTICS)

            if significant is None and p_value is not None:
                significant = p_value < self.alpha

            direction = self._direction(car)
            significance = self._significance(significant)
            strength = self._strength(car)
            label = self._label(direction, significance)
            alignment = self._alignment(sentiment_value, direction)

            return EventImpactRecord(
                event_id=event_id,
                ticker=ticker,
                sentiment=sentiment_value,
                effective_time=effective_time,
                realized_return=realized,
                abnormal_return=abnormal_return,
                car=car,
                statistic=statistic,
                p_value=p_value,
                significant=significant,
                impact_direction=direction,
                impact_strength=strength,
                statistical_significance=significance,
                sentiment_alignment=alignment,
                impact_label=label,
                status=ImpactStatus.OBSERVED,
            )
        except (TypeError, ValueError):
            return self._invalid(event_id, ticker, sentiment_value, effective_time, realized)

    def _record(self, event_id, ticker, sentiment, effective_time, realized, study, status):
        try:
            car = _phase6_car(study, self.car_window)
            significant = _phase6_significant(study) if study is not None else None
        except ValueError:
            return self._invalid(event_id, ticker, sentiment, effective_time, realized)
        direction = self._direction(car) if car is not None else ImpactDirection.UNKNOWN
        significance = self._significance(significant)
        return EventImpactRecord(
            event_id, ticker, sentiment, effective_time, realized,
            _phase6_abnormal_return(study),
            car,
            _phase6_statistic(study),
            _phase6_p_value(study),
            significant,
            direction,
            self._strength(car) if car is not None else ImpactStrength.UNKNOWN,
            significance,
            self._alignment(sentiment, direction) if sentiment else SentimentAlignment.UNKNOWN,
            self._label(direction, significance),
            status,
        )

    @staticmethod
    def _missing_sentiment_record(event_id, ticker, effective_time, realized):
        """Sentiment is required to form a canonical event-impact label.

        Preserve the observed realized return, but do not expose Phase-6
        impact statistics as an impact result when the event sentiment is
        absent. This prevents a partial record from being mistaken for a
        complete impact classification.
        """
        return EventImpactRecord(
            event_id=event_id,
            ticker=ticker,
            sentiment=None,
            effective_time=effective_time,
            realized_return=realized,
            abnormal_return=None,
            car=None,
            statistic=None,
            p_value=None,
            significant=None,
            impact_direction=ImpactDirection.UNKNOWN,
            impact_strength=ImpactStrength.UNKNOWN,
            statistical_significance=StatisticalSignificance.UNKNOWN,
            sentiment_alignment=SentimentAlignment.UNKNOWN,
            impact_label=ImpactLabel.UNKNOWN,
            status=ImpactStatus.MISSING_SENTIMENT,
        )

    @staticmethod
    def _direction(car):
        if car is None:
            return ImpactDirection.UNKNOWN
        if car > 0:
            return ImpactDirection.POSITIVE
        if car < 0:
            return ImpactDirection.NEGATIVE
        return ImpactDirection.NEUTRAL

    def _strength(self, car):
        magnitude = abs(car)
        if magnitude >= self.thresholds.high_abs_car:
            return ImpactStrength.HIGH
        if magnitude >= self.thresholds.medium_abs_car:
            return ImpactStrength.MEDIUM
        return ImpactStrength.LOW

    @staticmethod
    def _significance(significant):
        if significant is True:
            return StatisticalSignificance.SIGNIFICANT
        if significant is False:
            return StatisticalSignificance.INSIGNIFICANT
        return StatisticalSignificance.UNKNOWN

    @staticmethod
    def _label(direction, significance):
        if direction == ImpactDirection.NEUTRAL:
            return ImpactLabel.NEUTRAL
        if direction == ImpactDirection.POSITIVE:
            if significance == StatisticalSignificance.SIGNIFICANT:
                return ImpactLabel.POSITIVE_SIGNIFICANT
            if significance == StatisticalSignificance.INSIGNIFICANT:
                return ImpactLabel.POSITIVE_INSIGNIFICANT
        if direction == ImpactDirection.NEGATIVE:
            if significance == StatisticalSignificance.SIGNIFICANT:
                return ImpactLabel.NEGATIVE_SIGNIFICANT
            if significance == StatisticalSignificance.INSIGNIFICANT:
                return ImpactLabel.NEGATIVE_INSIGNIFICANT
        return ImpactLabel.UNKNOWN

    @staticmethod
    def _alignment(sentiment, direction):
        if direction == ImpactDirection.NEUTRAL:
            return SentimentAlignment.NEUTRAL
        if direction not in {ImpactDirection.POSITIVE, ImpactDirection.NEGATIVE}:
            return SentimentAlignment.UNKNOWN
        value = sentiment.upper()
        if value in {"POSITIVE", "POS", "1"}:
            return SentimentAlignment.ALIGNED if direction == ImpactDirection.POSITIVE else SentimentAlignment.CONTRADICTED
        if value in {"NEGATIVE", "NEG", "-1"}:
            return SentimentAlignment.ALIGNED if direction == ImpactDirection.NEGATIVE else SentimentAlignment.CONTRADICTED
        return SentimentAlignment.UNKNOWN

    @staticmethod
    def _invalid(event_id, ticker, sentiment, effective_time, realized):
        return EventImpactRecord(
            event_id, ticker, sentiment, effective_time, realized,
            None, None, None, None, None,
            ImpactDirection.UNKNOWN, ImpactStrength.UNKNOWN,
            StatisticalSignificance.UNKNOWN, SentimentAlignment.UNKNOWN,
            ImpactLabel.UNKNOWN, ImpactStatus.INVALID_INPUT,
        )
