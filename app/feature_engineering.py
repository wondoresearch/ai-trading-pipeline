"""Phase 10 — leakage-safe feature engineering.

Consumes Phase 8 research records and produces model-ready numeric features.
Only information present in each record's ``features`` mapping is used.
Targets and post-event outcomes are never read.

The transformer must be fitted on the training partition only. Categorical
vocabularies and numeric scaling parameters are therefore learned without
looking at validation/test observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import math
from typing import Any, Mapping, Sequence


class FeatureEngineeringStatus(str, Enum):
    READY = "READY"
    NOT_FITTED = "NOT_FITTED"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class FeatureVector:
    event_id: str
    ticker: str
    values: tuple[float, ...]
    feature_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "feature_names": list(self.feature_names),
            "values": list(self.values),
        }


@dataclass(frozen=True)
class FeatureDataset:
    status: FeatureEngineeringStatus
    feature_names: tuple[str, ...]
    records: tuple[FeatureVector, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "feature_names": list(self.feature_names),
            "records": [record.to_dict() for record in self.records],
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True)
class _NumericStats:
    mean: float
    std: float


class FeatureEngineer:
    """Fit deterministic, leakage-safe transformations on training features."""

    FORBIDDEN_TARGET_KEYS = frozenset({
        "event_day_return", "t1_return", "t3_return", "t5_return", "t10_return",
        "abnormal_return", "car", "car_-1_1", "car_0_1", "car_0_3", "car_0_5",
        "car_0_10", "aar", "caar", "statistic", "p_value", "significant",
        "impact_direction", "impact_strength", "statistical_significance",
        "sentiment_alignment", "impact_label",
    })

    DEFAULT_NUMERIC = (
        "sentiment_score",
        "sentiment_magnitude",
    )
    DEFAULT_CATEGORICAL = (
        "event_type",
        "sentiment_label",
    )

    def __init__(
        self,
        *,
        numeric_fields: Sequence[str] = DEFAULT_NUMERIC,
        categorical_fields: Sequence[str] = DEFAULT_CATEGORICAL,
        scale_numeric: bool = True,
    ) -> None:
        self.numeric_fields = tuple(dict.fromkeys(numeric_fields))
        self.categorical_fields = tuple(dict.fromkeys(categorical_fields))
        self.scale_numeric = scale_numeric
        self._stats: dict[str, _NumericStats] = {}
        self._vocab: dict[str, tuple[str, ...]] = {}
        self._feature_names: tuple[str, ...] = ()
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def fit(self, training_dataset: Any) -> "FeatureEngineer":
        records = self._records(training_dataset)
        if not records:
            raise ValueError("training dataset must contain at least one record")

        for record in records:
            if self._has_leakage(record["features"]):
                raise ValueError(
                    f"forbidden target field found in features for "
                    f"{record['event_id']}:{record['ticker']}"
                )

        stats: dict[str, _NumericStats] = {}
        for field in self.numeric_fields:
            values = self._numeric_values(
                record["features"].get(field) for record in records
            )
            if values:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                std = math.sqrt(variance)
                stats[field] = _NumericStats(mean, std if std > 0 else 1.0)

        vocab: dict[str, tuple[str, ...]] = {}
        for field in self.categorical_fields:
            values = {
                str(record["features"][field])
                for record in records
                if record["features"].get(field) is not None
            }
            vocab[field] = tuple(sorted(values))

        names: list[str] = []
        for field in self.numeric_fields:
            names.append(field)
        for field in self.categorical_fields:
            names.extend(f"{field}={value}" for value in vocab[field])
        names.extend([
            "effective_hour_sin",
            "effective_hour_cos",
            "effective_weekday_sin",
            "effective_weekday_cos",
        ])

        self._stats = stats
        self._vocab = vocab
        self._feature_names = tuple(names)
        self._fitted = True
        return self

    def transform(self, dataset: Any) -> FeatureDataset:
        if not self._fitted:
            return FeatureDataset(
                FeatureEngineeringStatus.NOT_FITTED,
                (),
                (),
                "FeatureEngineer.fit() must be called on training data first.",
            )

        try:
            records = self._records(dataset)
            vectors = tuple(self._transform_record(record) for record in records)
        except (TypeError, ValueError) as exc:
            return FeatureDataset(
                FeatureEngineeringStatus.INVALID_INPUT,
                self._feature_names,
                (),
                str(exc),
            )

        return FeatureDataset(
            FeatureEngineeringStatus.READY,
            self._feature_names,
            vectors,
        )

    def fit_transform(self, training_dataset: Any) -> FeatureDataset:
        self.fit(training_dataset)
        return self.transform(training_dataset)

    def _transform_record(self, record: Mapping[str, Any]) -> FeatureVector:
        features = record["features"]
        if self._has_leakage(features):
            raise ValueError(
                f"forbidden target field found in features for "
                f"{record['event_id']}:{record['ticker']}"
            )

        values: list[float] = []
        for field in self.numeric_fields:
            raw = features.get(field)
            if raw is None:
                values.append(0.0)
                continue
            try:
                number = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-numeric value in {field}") from exc
            if not math.isfinite(number):
                raise ValueError(f"non-finite value in {field}")
            if self.scale_numeric and field in self._stats:
                stat = self._stats[field]
                number = (number - stat.mean) / stat.std
            values.append(number)

        for field in self.categorical_fields:
            current = features.get(field)
            for value in self._vocab[field]:
                values.append(1.0 if current is not None and str(current) == value else 0.0)

        effective_time = self._parse_time(features.get("effective_time"))
        if effective_time is None:
            raise ValueError("effective_time must be timezone-aware and parseable")

        hour = effective_time.hour + effective_time.minute / 60.0
        hour_angle = 2.0 * math.pi * hour / 24.0
        weekday_angle = 2.0 * math.pi * effective_time.weekday() / 7.0
        values.extend([
            math.sin(hour_angle),
            math.cos(hour_angle),
            math.sin(weekday_angle),
            math.cos(weekday_angle),
        ])

        if not all(math.isfinite(value) for value in values):
            raise ValueError("feature transformation produced a non-finite value")

        return FeatureVector(
            event_id=str(record["event_id"]),
            ticker=str(record["ticker"]),
            values=tuple(values),
            feature_names=self._feature_names,
        )

    @classmethod
    def _records(cls, dataset: Any) -> list[dict[str, Any]]:
        records = getattr(dataset, "records", dataset)
        result = []
        for record in records:
            if isinstance(record, Mapping):
                features = dict(record.get("features") or {})
                event_id = record.get("event_id") or features.get("event_id")
                ticker = record.get("ticker") or features.get("ticker")
            else:
                features = dict(getattr(record, "features", {}) or {})
                event_id = getattr(record, "event_id", None) or features.get("event_id")
                ticker = getattr(record, "ticker", None) or features.get("ticker")
            if event_id is None or ticker is None:
                raise ValueError("every record requires event_id and ticker")
            result.append({
                "event_id": event_id,
                "ticker": ticker,
                "features": features,
            })
        return result

    @classmethod
    def _has_leakage(cls, features: Mapping[str, Any]) -> bool:
        return bool(set(features) & cls.FORBIDDEN_TARGET_KEYS)

    @staticmethod
    def _numeric_values(values: Any) -> list[float]:
        result = []
        for value in values:
            if value is None or isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                result.append(number)
        return result

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed
