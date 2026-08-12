"""Phase 8 canonical research dataset and feature/target separation.

Consumes Phase 1-7 outputs only. It does not fetch market data, recalculate
returns, rerun event-study statistics, or recreate impact labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import json
import math
from typing import Any, Iterable, Mapping, Optional, Sequence


class ResearchDatasetStatus(str, Enum):
    OBSERVED = "observed"
    PARTIAL = "partial"
    INVALID = "invalid"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class ResearchRecordInput:
    """One canonical event×ticker bundle from Phase 1-7."""

    event: Any
    event_time: Any
    price_observation: Any
    return_result: Any = None
    event_study: Any = None
    event_impact: Any = None
    cross_sectional: Any = None


@dataclass(frozen=True)
class ResearchRecord:
    event_id: str
    ticker: str
    status: ResearchDatasetStatus
    features: Mapping[str, Any]
    targets: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "status": self.status.value,
            "features": dict(self.features),
            "targets": dict(self.targets),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResearchDataset:
    dataset_version: str
    schema_version: str
    generated_at: str
    records: tuple[ResearchRecord, ...]

    def features(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(record.features for record in self.records)

    def targets(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(record.targets for record in self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class ResearchDatasetBuilder:
    """Build the Phase 8 canonical event×ticker research dataset."""

    DATASET_VERSION = "phase8-v1"
    SCHEMA_VERSION = "research-event-v1"
    TARGET_KEYS = frozenset(
        {
            "event_day_return",
            "t1_return",
            "t3_return",
            "t5_return",
            "t10_return",
            "car",
            "car_-1_1",
            "car_0_1",
            "car_0_3",
            "car_0_5",
            "car_0_10",
            "abnormal_return",
            "aar",
            "caar",
            "statistic",
            "p_value",
            "significant",
            "impact_direction",
            "impact_strength",
            "statistical_significance",
            "sentiment_alignment",
            "impact_label",
        }
    )

    def build(
        self,
        inputs: Iterable[ResearchRecordInput],
        *,
        generated_at: Optional[datetime] = None,
    ) -> ResearchDataset:
        records = [self.build_record(item) for item in inputs]
        records.sort(key=lambda item: (item.features["effective_time"], item.event_id, item.ticker))

        seen: set[tuple[str, str]] = set()
        deduped: list[ResearchRecord] = []
        duplicate_found = False
        for record in records:
            key = (record.event_id, record.ticker)
            if key in seen:
                duplicate_found = True
                deduped.append(
                    ResearchRecord(
                        record.event_id,
                        record.ticker,
                        ResearchDatasetStatus.DUPLICATE,
                        record.features,
                        {key: None for key in record.targets},
                        dict(record.metadata),
                    )
                )
            else:
                seen.add(key)
                deduped.append(record)

        if duplicate_found:
            # Duplicate records remain visible and explicit; nothing is silently overwritten.
            pass

        timestamp = generated_at or datetime.now().astimezone()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return ResearchDataset(
            self.DATASET_VERSION,
            self.SCHEMA_VERSION,
            timestamp.isoformat(),
            tuple(deduped),
        )

    def build_record(self, item: ResearchRecordInput) -> ResearchRecord:
        event = item.event
        resolution = item.event_time
        observation = item.price_observation

        event_id = self._first(
            event, "event_id", default=self._first(resolution, "event_id", default=self._first(observation, "event_id"))
        )
        ticker = self._first(
            event, "ticker", default=self._first(resolution, "ticker", default=self._first(observation, "ticker"))
        )
        effective_time = self._first(resolution, "effective_time", default=self._first(observation, "effective_time"))

        if not event_id or not ticker or not self._is_aware_datetime(effective_time):
            raise ValueError("Phase 8 requires event_id, ticker, and timezone-aware effective_time")

        # Cross-phase identity must agree before a record can be constructed.
        for source in (item.price_observation, item.return_result, item.event_study, item.event_impact):
            self._validate_identity(source, event_id, ticker, effective_time)

        effective_iso = effective_time.isoformat()
        market_session = self._enum_value(self._first(resolution, "market_session"))
        resolution_rule = self._enum_value(self._first(resolution, "resolution_rule"))
        event_date = self._first(
            item.event_study,
            "event_date",
            default=self._first(
                observation,
                "event_date",
                default=effective_time.date(),
            ),
        )
        event_day = self._date_string(event_date)

        sentiment_score = self._first(event, "sentiment_score")
        sentiment_label = self._first(event, "sentiment_label", default=self._first(event, "sentiment"))
        event_type = self._first(event, "event_type")
        published_at = self._first(
            event,
            "published_at_utc",
            default=self._first(event, "published_at"),
        )

        features = {
            "event_id": event_id,
            "ticker": ticker,
            "published_at": self._iso_or_value(published_at),
            "effective_time": effective_iso,
            "event_day": event_day,
            "event_type": event_type,
            "event_session": market_session,
            "resolution_rule": resolution_rule,
            "is_trading_day": self._first(resolution, "is_trading_day"),
            "is_tradeable_at_event": self._first(resolution, "is_tradeable_at_event"),
            "is_same_session_effective": self._first(resolution, "is_same_session_effective"),
            "sentiment_score": self._numeric_or_none(sentiment_score),
            "sentiment_label": self._enum_value(sentiment_label),
        }

        targets = self._build_targets(item.return_result, item.event_study, item.event_impact, item.cross_sectional)

        missing_feature_keys = {
            "published_at",
            "event_type",
            "sentiment_score",
            "sentiment_label",
        }
        partial = any(features[key] is None for key in missing_feature_keys)

        # Individual-record completeness is based only on canonical per-event
        # outcomes. AAR/CAAR and inference are cohort-level outputs and must not
        # make an otherwise complete event×ticker record partial.
        required_outcomes = {
            "event_day_return",
            "t1_return",
            "t3_return",
            "t5_return",
            "t10_return",
            "car",
            "impact_label",
        }
        if any(targets[key] is None for key in required_outcomes):
            partial = True

        source_statuses = {
            "price_observation_status": self._enum_value(self._first(observation, "status")),
            "return_status": self._enum_value(self._first(item.return_result, "status")),
            "event_study_status": self._enum_value(self._first(item.event_study, "status")),
            "event_impact_status": self._enum_value(self._first(item.event_impact, "status")),
        }

        metadata = {
            "source_phases": (
                "phase_1",
                "phase_2",
                "phase_3",
                "phase_4",
                "phase_5",
                "phase_6",
                "phase_7",
            ),
            "price_source": self._first(observation, "price_source"),
            "price_granularity": self._first(observation, "price_granularity"),
            **source_statuses,
        }

        status = ResearchDatasetStatus.PARTIAL if partial else ResearchDatasetStatus.OBSERVED
        return ResearchRecord(event_id, ticker, status, features, targets, metadata)

    def _build_targets(self, return_result: Any, event_study: Any, impact: Any, cross_sectional: Any) -> dict[str, Any]:
        targets: dict[str, Any] = {
            "event_day_return": self._return_value(return_result, "event_day_return", 0),
            "t1_return": self._return_value(return_result, "t1_return", 1),
            "t3_return": self._return_value(return_result, "t3_return", 3),
            "t5_return": self._return_value(return_result, "t5_return", 5),
            "t10_return": self._return_value(return_result, "t10_return", 10),
            "car": self._window_value(event_study, "car_0_10", "car"),
            "car_-1_1": self._window_value(event_study, "car_-1_1", "car"),
            "car_0_1": self._window_value(event_study, "car_0_1", "car"),
            "car_0_3": self._window_value(event_study, "car_0_3", "car"),
            "car_0_5": self._window_value(event_study, "car_0_5", "car"),
            "car_0_10": self._window_value(event_study, "car_0_10", "car"),
            "abnormal_return": self._first(impact, "abnormal_return"),
            "aar": self._first(cross_sectional, "aar"),
            "caar": self._first(cross_sectional, "caar"),
            "statistic": self._first(impact, "statistic", default=self._nested(cross_sectional, "inference", "statistic")),
            "p_value": self._first(impact, "p_value", default=self._nested(cross_sectional, "inference", "p_value")),
            "significant": self._first(impact, "significant", default=self._first(impact, "statistical_significance")),
            "impact_direction": self._enum_value(self._first(impact, "impact_direction")),
            "impact_strength": self._enum_value(self._first(impact, "impact_strength")),
            "statistical_significance": self._enum_value(self._first(impact, "statistical_significance")),
            "sentiment_alignment": self._enum_value(self._first(impact, "sentiment_alignment")),
            "impact_label": self._enum_value(self._first(impact, "impact_label")),
        }
        return {key: self._json_scalar(value) for key, value in targets.items()}

    @classmethod
    def _return_value(cls, result: Any, key: str, offset: int) -> Any:
        sentinel = object()
        direct = cls._first(result, key, default=sentinel)
        if direct is not sentinel:
            # A present-but-null canonical field is authoritative; do not
            # replace missing data from a secondary representation.
            return cls._json_scalar(direct)

        for container_name in ("forward_returns", "forward_return", "cumulative_returns", "cumulative_return"):
            container = cls._first(result, container_name)
            if isinstance(container, Mapping):
                for lookup in (key, key.replace("_return", ""), f"t{offset}"):
                    if lookup in container:
                        return cls._json_scalar(container[lookup])
        return None

    @classmethod
    def _window_value(cls, study: Any, window_name: str, field: str) -> Any:
        windows = cls._first(study, "windows", default=())
        if isinstance(windows, Mapping):
            candidate = windows.get(window_name)
            return cls._json_scalar(cls._first(candidate, field))
        for window in windows or ():
            if cls._first(window, "name") == window_name:
                return cls._json_scalar(cls._first(window, field))
        return None

    @staticmethod
    def _nested(source: Any, *keys: str) -> Any:
        current = source
        for key in keys:
            current = ResearchDatasetBuilder._first(current, key)
            if current is None:
                return None
        return current

    @staticmethod
    def _first(source: Any, key: str, default: Any = None) -> Any:
        if source is None:
            return default
        if isinstance(source, Mapping):
            return source.get(key, default)
        return getattr(source, key, default)

    @classmethod
    def _validate_identity(cls, source: Any, event_id: str, ticker: str, effective_time: datetime) -> None:
        if source is None:
            return
        source_event_id = cls._first(source, "event_id")
        source_ticker = cls._first(source, "ticker")
        source_effective = cls._first(source, "effective_time")
        if source_event_id is not None and source_event_id != event_id:
            raise ValueError("Phase 8 identity mismatch: event_id")
        if source_ticker is not None and source_ticker != ticker:
            raise ValueError("Phase 8 identity mismatch: ticker")
        if source_effective is not None:
            if not isinstance(source_effective, datetime):
                raise ValueError("Phase 8 effective_time must be datetime")
            if source_effective != effective_time:
                raise ValueError("Phase 8 identity mismatch: effective_time")

    @staticmethod
    def _is_aware_datetime(value: Any) -> bool:
        return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None

    @staticmethod
    def _enum_value(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        return value

    @staticmethod
    def _date_string(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return value

    @staticmethod
    def _iso_or_value(value: Any) -> Any:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("published_at must be timezone-aware")
            return value.isoformat()
        return value

    @staticmethod
    def _numeric_or_none(value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @classmethod
    def _json_scalar(cls, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, Mapping):
            return {str(k): cls._json_scalar(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (tuple, list)):
            return [cls._json_scalar(v) for v in value]
        return value
