"""Phase 9 dataset validation and ML-readiness checks.

Validates the canonical Phase 8 research dataset without modifying it.
No model training, feature engineering, market-data fetching, or outcome
recalculation is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import math
from typing import Any, Iterable, Mapping, Optional, Sequence


class ValidationStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class ValidationIssue:
    check: str
    status: ValidationStatus
    message: str
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "status": self.status.value,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class TimeSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    boundaries: Mapping[str, Optional[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "boundaries": dict(self.boundaries),
        }


@dataclass(frozen=True)
class ValidationReport:
    status: ValidationStatus
    sample_size: int
    checks: tuple[ValidationIssue, ...]
    feature_summary: Mapping[str, Any]
    target_summary: Mapping[str, Any]
    time_split: Optional[TimeSplit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "sample_size": self.sample_size,
            "checks": [item.to_dict() for item in self.checks],
            "feature_summary": dict(self.feature_summary),
            "target_summary": dict(self.target_summary),
            "time_split": self.time_split.to_dict() if self.time_split else None,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class DatasetValidator:
    """Validate Phase 8 data for downstream ML/research use."""

    DEFAULT_FORBIDDEN_FEATURES = frozenset({
        "event_day_return", "t1_return", "t3_return", "t5_return", "t10_return",
        "abnormal_return", "car", "car_-1_1", "car_0_1", "car_0_3", "car_0_5",
        "car_0_10", "aar", "caar", "statistic", "p_value", "significant",
        "impact_direction", "impact_strength", "statistical_significance",
        "sentiment_alignment", "impact_label",
    })

    def validate(
        self,
        dataset: Any,
        *,
        target: str = "impact_label",
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        min_class_count: int = 5,
        outlier_z_threshold: float = 4.0,
    ) -> ValidationReport:
        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio must be between 0 and 1")
        if not (0 <= validation_ratio < 1):
            raise ValueError("validation_ratio must be between 0 and 1")
        if train_ratio + validation_ratio >= 1:
            raise ValueError("train_ratio + validation_ratio must be < 1")
        if min_class_count < 1:
            raise ValueError("min_class_count must be >= 1")
        if outlier_z_threshold <= 0:
            raise ValueError("outlier_z_threshold must be > 0")

        records = list(self._records(dataset))
        checks: list[ValidationIssue] = []

        checks.append(self._leakage_check(records))
        checks.append(self._duplicate_check(records))
        checks.append(self._feature_schema_check(records))
        checks.append(self._datetime_check(records))

        feature_summary = self._summarize_features(records)
        target_summary = self._summarize_target(records, target, min_class_count)

        checks.append(self._missingness_check(records, feature_summary))
        checks.append(self._target_check(target_summary, target))
        checks.append(self._class_balance_check(target_summary, min_class_count))
        checks.append(self._outlier_check(records, outlier_z_threshold))

        split, split_issue = self._time_split(
            records, train_ratio, validation_ratio
        )
        checks.append(split_issue)

        status = ValidationStatus.PASS
        if any(item.status == ValidationStatus.FAIL for item in checks):
            status = ValidationStatus.FAIL
        elif any(item.status == ValidationStatus.WARN for item in checks):
            status = ValidationStatus.WARN

        return ValidationReport(
            status=status,
            sample_size=len(records),
            checks=tuple(checks),
            feature_summary=feature_summary,
            target_summary=target_summary,
            time_split=split,
        )

    @staticmethod
    def _records(dataset: Any) -> Iterable[Mapping[str, Any]]:
        records = getattr(dataset, "records", dataset)
        for record in records:
            if isinstance(record, Mapping):
                features = record.get("features", {})
                targets = record.get("targets", {})
                yield {
                    "event_id": record.get("event_id"),
                    "ticker": record.get("ticker"),
                    "status": record.get("status"),
                    "features": dict(features or {}),
                    "targets": dict(targets or {}),
                }
            else:
                yield {
                    "event_id": getattr(record, "event_id", None),
                    "ticker": getattr(record, "ticker", None),
                    "status": getattr(getattr(record, "status", None), "value", getattr(record, "status", None)),
                    "features": dict(getattr(record, "features", {}) or {}),
                    "targets": dict(getattr(record, "targets", {}) or {}),
                }

    @classmethod
    def _leakage_check(cls, records: Sequence[Mapping[str, Any]]) -> ValidationIssue:
        leaked: list[str] = []
        for index, record in enumerate(records):
            overlap = set(record["features"]) & cls.DEFAULT_FORBIDDEN_FEATURES
            if overlap:
                leaked.extend(f"{index}:{key}" for key in sorted(overlap))
        if leaked:
            return ValidationIssue(
                "leakage_detection", ValidationStatus.FAIL,
                "Future outcome fields are present in the feature layer.",
                {"leaked_fields": leaked[:100], "count": len(leaked)},
            )
        return ValidationIssue(
            "leakage_detection", ValidationStatus.PASS,
            "No canonical Phase 5-7 outcome fields were found in features.",
            {},
        )

    @staticmethod
    def _duplicate_check(records: Sequence[Mapping[str, Any]]) -> ValidationIssue:
        seen: set[tuple[Any, Any]] = set()
        duplicates: list[tuple[Any, Any]] = []
        for record in records:
            key = (record["event_id"], record["ticker"])
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        if duplicates:
            return ValidationIssue(
                "duplicate_analysis", ValidationStatus.FAIL,
                "Duplicate event_id+ticker keys exist.",
                {"duplicates": [list(x) for x in duplicates[:100]], "count": len(duplicates)},
            )
        return ValidationIssue("duplicate_analysis", ValidationStatus.PASS,
                               "No duplicate event_id+ticker keys found.", {})

    @staticmethod
    def _feature_schema_check(records: Sequence[Mapping[str, Any]]) -> ValidationIssue:
        missing: list[str] = []
        required = ("event_id", "ticker", "effective_time")
        for index, record in enumerate(records):
            for key in required:
                if record["features"].get(key) is None and record.get(key) is None:
                    missing.append(f"{index}:{key}")
        if missing:
            return ValidationIssue(
                "feature_schema", ValidationStatus.FAIL,
                "Required identity/time fields are missing.",
                {"missing": missing[:100], "count": len(missing)},
            )
        return ValidationIssue("feature_schema", ValidationStatus.PASS,
                               "Required identity/time fields are present.", {})

    @staticmethod
    def _datetime_check(records: Sequence[Mapping[str, Any]]) -> ValidationIssue:
        invalid = []
        for index, record in enumerate(records):
            value = record["features"].get("effective_time")
            if value is None:
                continue
            if not DatasetValidator._parse_time(value):
                invalid.append(index)
        if invalid:
            return ValidationIssue(
                "time_order_validation", ValidationStatus.FAIL,
                "One or more effective_time values are invalid or timezone-naive.",
                {"record_indexes": invalid[:100], "count": len(invalid)},
            )
        return ValidationIssue("time_order_validation", ValidationStatus.PASS,
                               "All effective_time values are timezone-aware and parseable.", {})

    @staticmethod
    def _summarize_features(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        keys = sorted({key for r in records for key in r["features"]})
        result: dict[str, Any] = {}
        for key in keys:
            values = [r["features"].get(key) for r in records]
            non_null = [v for v in values if v is not None]
            numeric = DatasetValidator._numeric_values(non_null)
            result[key] = {
                "count": len(values),
                "non_null": len(non_null),
                "missing": len(values) - len(non_null),
                "numeric": bool(numeric) and len(numeric) == len(non_null),
                "min": min(numeric) if numeric and len(numeric) == len(non_null) else None,
                "max": max(numeric) if numeric and len(numeric) == len(non_null) else None,
            }
        return result

    @staticmethod
    def _summarize_target(records: Sequence[Mapping[str, Any]], target: str, min_class_count: int) -> dict[str, Any]:
        values = [r["targets"].get(target) for r in records]
        present = [v for v in values if v is not None]
        numeric = DatasetValidator._numeric_values(present)
        counts: dict[str, int] = {}
        if not numeric or len(numeric) != len(present):
            for value in present:
                counts[str(value)] = counts.get(str(value), 0) + 1
        return {
            "name": target,
            "count": len(values),
            "non_null": len(present),
            "missing": len(values) - len(present),
            "numeric": bool(present) and len(numeric) == len(present),
            "class_counts": dict(sorted(counts.items())),
            "min_class_count": min(counts.values()) if counts else None,
            "numeric_min": min(numeric) if numeric and len(numeric) == len(present) else None,
            "numeric_max": max(numeric) if numeric and len(numeric) == len(present) else None,
        }

    @staticmethod
    def _missingness_check(records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> ValidationIssue:
        high = {
            key: data["missing"]
            for key, data in summary.items()
            if data["count"] and data["missing"] / data["count"] > 0.20
        }
        if high:
            return ValidationIssue(
                "missing_data_analysis", ValidationStatus.WARN,
                "One or more features have more than 20% missing values.",
                {"fields": high},
            )
        return ValidationIssue("missing_data_analysis", ValidationStatus.PASS,
                               "No feature exceeds the 20% missingness warning threshold.", {})

    @staticmethod
    def _target_check(summary: Mapping[str, Any], target: str) -> ValidationIssue:
        if summary["non_null"] == 0:
            return ValidationIssue("target_validation", ValidationStatus.FAIL,
                                   "Target contains no usable observations.", {"target": target})
        return ValidationIssue("target_validation", ValidationStatus.PASS,
                               "Target contains usable observations.", {
                                   "target": target,
                                   "non_null": summary["non_null"],
                               })

    @staticmethod
    def _class_balance_check(summary: Mapping[str, Any], min_class_count: int) -> ValidationIssue:
        counts = summary["class_counts"]
        if not counts:
            return ValidationIssue("class_balance", ValidationStatus.PASS,
                                   "Target is numeric; class-balance check is not applicable.", {})
        rare = {key: value for key, value in counts.items() if value < min_class_count}
        if rare:
            return ValidationIssue(
                "class_balance", ValidationStatus.WARN,
                "One or more classes have fewer observations than the minimum recommended count.",
                {"minimum": min_class_count, "rare_classes": rare, "class_counts": counts},
            )
        return ValidationIssue("class_balance", ValidationStatus.PASS,
                               "All target classes meet the minimum recommended count.",
                               {"class_counts": counts})

    @staticmethod
    def _outlier_check(records: Sequence[Mapping[str, Any]], threshold: float) -> ValidationIssue:
        """Detect row-level numeric feature outliers using a deterministic z-score.

        A field with fewer than 3 numeric observations is not scored. Constant
        fields have zero variance and therefore have no z-score outliers.
        Missing/non-numeric values are ignored for the field's calculation.
        """
        field_values: dict[str, list[tuple[int, float]]] = {}
        for index, record in enumerate(records):
            for key, value in record["features"].items():
                if key in DatasetValidator.DEFAULT_FORBIDDEN_FEATURES:
                    continue
                if isinstance(value, bool):
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(number):
                    continue
                field_values.setdefault(key, []).append((index, number))

        outliers: dict[str, list[int]] = {}
        numeric_fields: list[str] = []
        for key in sorted(field_values):
            observations = field_values[key]
            if len(observations) < 3:
                continue
            numeric_fields.append(key)
            mean = sum(value for _, value in observations) / len(observations)
            variance = sum((value - mean) ** 2 for _, value in observations) / len(observations)
            std = math.sqrt(variance)
            if std == 0:
                continue
            indexes = [
                index
                for index, value in observations
                if abs((value - mean) / std) > threshold
            ]
            if indexes:
                outliers[key] = indexes

        if outliers:
            return ValidationIssue(
                "outlier_detection",
                ValidationStatus.WARN,
                "Numeric features contain observations beyond the configured z-score threshold.",
                {
                    "numeric_fields": numeric_fields,
                    "z_threshold": threshold,
                    "outliers": outliers,
                    "outlier_count": sum(len(v) for v in outliers.values()),
                },
            )

        return ValidationIssue(
            "outlier_detection",
            ValidationStatus.PASS,
            "No numeric feature observations exceed the configured z-score threshold.",
            {
                "numeric_fields": numeric_fields,
                "z_threshold": threshold,
                "outlier_count": 0,
            },
        )

    @classmethod
    def _time_split(
        cls,
        records: Sequence[Mapping[str, Any]],
        train_ratio: float,
        validation_ratio: float,
    ) -> tuple[Optional[TimeSplit], ValidationIssue]:
        indexed = []
        for index, record in enumerate(records):
            parsed = cls._parse_time(record["features"].get("effective_time"))
            if parsed is None:
                return None, ValidationIssue(
                    "time_based_split", ValidationStatus.FAIL,
                    "Cannot create a chronological split because effective_time is invalid.",
                    {},
                )
            indexed.append((parsed, index))
        indexed.sort(key=lambda item: (item[0], item[1]))
        n = len(indexed)
        if n < 3:
            return None, ValidationIssue(
                "time_based_split", ValidationStatus.WARN,
                "At least three records are recommended for train/validation/test splitting.",
                {"sample_size": n},
            )
        train_end = max(1, int(n * train_ratio))
        val_end = max(train_end + 1, int(n * (train_ratio + validation_ratio)))
        if val_end >= n:
            val_end = n - 1
        if train_end >= val_end:
            return None, ValidationIssue(
                "time_based_split", ValidationStatus.FAIL,
                "Ratios do not produce non-empty train, validation, and test partitions.",
                {"sample_size": n},
            )
        train = tuple(index for _, index in indexed[:train_end])
        validation = tuple(index for _, index in indexed[train_end:val_end])
        test = tuple(index for _, index in indexed[val_end:])
        boundaries = {
            "train_end": indexed[train_end - 1][0].isoformat(),
            "validation_end": indexed[val_end - 1][0].isoformat(),
            "test_start": indexed[val_end][0].isoformat(),
        }
        return (
            TimeSplit(train, validation, test, boundaries),
            ValidationIssue(
                "time_based_split", ValidationStatus.PASS,
                "Chronological train/validation/test split created without shuffling.",
                {"train_size": len(train), "validation_size": len(validation), "test_size": len(test)},
            ),
        )

    @staticmethod
    def _numeric_values(values: Sequence[Any]) -> list[float]:
        result = []
        for value in values:
            if isinstance(value, bool):
                return []
            try:
                number = float(value)
            except (TypeError, ValueError):
                return []
            if not math.isfinite(number):
                return []
            result.append(number)
        return result

    @staticmethod
    def _parse_time(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                return None
            return value
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                return None
            return parsed
        return None
