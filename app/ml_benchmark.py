"""Phase 11 — reproducible ML benchmarking.

Benchmarks classification models using the Phase 9 chronological split and
the Phase 10 feature representation. Model selection uses validation metrics;
the test set is evaluated only after selection.

This module does not engineer features, resplit data, or use future outcomes
as inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Mapping, Sequence

try:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        average_precision_score,
        roc_auc_score,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("Phase 11 requires scikit-learn") from exc


class BenchmarkStatus(str, Enum):
    READY = "READY"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class SplitMetrics:
    sample_size: int
    accuracy: float | None
    balanced_accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    roc_auc: float | None
    pr_auc: float | None
    confusion_matrix: tuple[tuple[int, ...], ...]
    labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class ModelBenchmark:
    model_name: str
    parameters: Mapping[str, Any]
    validation: SplitMetrics
    test: SplitMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "parameters": dict(self.parameters),
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
        }


@dataclass(frozen=True)
class BenchmarkReport:
    status: BenchmarkStatus
    target: str
    selected_model: str | None
    feature_names: tuple[str, ...]
    train_size: int
    validation_size: int
    test_size: int
    class_distribution: Mapping[str, Mapping[str, int]]
    models: tuple[ModelBenchmark, ...]
    selection_metric: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "target": self.target,
            "selected_model": self.selected_model,
            "feature_names": list(self.feature_names),
            "train_size": self.train_size,
            "validation_size": self.validation_size,
            "test_size": self.test_size,
            "class_distribution": {
                key: dict(value) for key, value in self.class_distribution.items()
            },
            "models": [model.to_dict() for model in self.models],
            "selection_metric": self.selection_metric,
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


class MLBenchmark:
    """Benchmark baseline classifiers on an existing chronological split."""

    MODEL_FACTORIES = {
        "dummy": lambda: DummyClassifier(strategy="prior"),
        "logistic_regression": lambda: LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=1, class_weight="balanced"
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
    }

    def run(
        self,
        *,
        x_train: Sequence[Sequence[float]],
        y_train: Sequence[Any],
        x_validation: Sequence[Sequence[float]],
        y_validation: Sequence[Any],
        x_test: Sequence[Sequence[float]],
        y_test: Sequence[Any],
        feature_names: Sequence[str],
        target: str = "impact_label",
        selection_metric: str = "f1",
        models: Sequence[str] | None = None,
    ) -> BenchmarkReport:
        try:
            self._validate_inputs(
                x_train, y_train, x_validation, y_validation,
                x_test, y_test, feature_names, selection_metric
            )
        except ValueError as exc:
            return BenchmarkReport(
                BenchmarkStatus.INVALID_INPUT, target, None, tuple(feature_names),
                len(y_train), len(y_validation), len(y_test), {},
                (), selection_metric, str(exc)
            )

        requested = tuple(models or self.MODEL_FACTORIES)
        unknown = [name for name in requested if name not in self.MODEL_FACTORIES]
        if unknown:
            return BenchmarkReport(
                BenchmarkStatus.INVALID_INPUT, target, None, tuple(feature_names),
                len(y_train), len(y_validation), len(y_test), {},
                (), selection_metric, f"unknown models: {unknown}"
            )

        class_distribution = {
            "train": self._counts(y_train),
            "validation": self._counts(y_validation),
            "test": self._counts(y_test),
        }

        results: list[ModelBenchmark] = []
        for name in requested:
            model = self.MODEL_FACTORIES[name]()
            model.fit(x_train, y_train)
            validation = self._metrics(model, x_validation, y_validation)
            test = self._metrics(model, x_test, y_test)
            results.append(
                ModelBenchmark(
                    name,
                    self._parameters(model),
                    validation,
                    test,
                )
            )

        selected = max(
            results,
            key=lambda result: (
                self._metric_value(result.validation, selection_metric),
                -requested.index(result.model_name),
            ),
        ).model_name

        return BenchmarkReport(
            BenchmarkStatus.READY,
            target,
            selected,
            tuple(feature_names),
            len(y_train),
            len(y_validation),
            len(y_test),
            class_distribution,
            tuple(results),
            selection_metric,
        )

    @staticmethod
    def _validate_inputs(
        x_train, y_train, x_validation, y_validation,
        x_test, y_test, feature_names, selection_metric
    ) -> None:
        if selection_metric not in {"f1", "balanced_accuracy", "pr_auc", "roc_auc"}:
            raise ValueError("unsupported selection metric")
        if not y_train or not y_validation or not y_test:
            raise ValueError("train, validation, and test must all be non-empty")
        if len(x_train) != len(y_train):
            raise ValueError("train X/y lengths differ")
        if len(x_validation) != len(y_validation):
            raise ValueError("validation X/y lengths differ")
        if len(x_test) != len(y_test):
            raise ValueError("test X/y lengths differ")
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        width = len(feature_names)
        for matrix in (x_train, x_validation, x_test):
            if any(len(row) != width for row in matrix):
                raise ValueError("feature matrix width differs from feature_names")

    @staticmethod
    def _counts(values: Sequence[Any]) -> dict[str, int]:
        result: dict[str, int] = {}
        for value in values:
            key = str(value)
            result[key] = result.get(key, 0) + 1
        return dict(sorted(result.items()))

    @classmethod
    def _metrics(cls, model, x, y) -> SplitMetrics:
        labels = tuple(sorted({str(value) for value in y}))
        predicted = model.predict(x)
        all_labels = sorted({*map(str, y), *map(str, predicted)})
        matrix = confusion_matrix(
            [str(value) for value in y],
            [str(value) for value in predicted],
            labels=all_labels,
        )

        accuracy = accuracy_score(y, predicted)
        balanced = balanced_accuracy_score(y, predicted)
        precision = precision_score(y, predicted, average="weighted", zero_division=0)
        recall = recall_score(y, predicted, average="weighted", zero_division=0)
        f1 = f1_score(y, predicted, average="weighted", zero_division=0)

        roc_auc = None
        pr_auc = None
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(x)
            except Exception:
                probabilities = None

        if probabilities is not None and len(set(y)) == 2:
            positive_class = model.classes_[1]
            positive = [1.0 if value == positive_class else 0.0 for value in y]
            score = probabilities[:, 1]
            try:
                roc_auc = float(roc_auc_score(positive, score))
                pr_auc = float(average_precision_score(positive, score))
            except ValueError:
                pass

        return SplitMetrics(
            len(y),
            cls._finite_or_none(accuracy),
            cls._finite_or_none(balanced),
            cls._finite_or_none(precision),
            cls._finite_or_none(recall),
            cls._finite_or_none(f1),
            roc_auc,
            pr_auc,
            tuple(tuple(int(value) for value in row) for row in matrix),
            tuple(all_labels),
        )

    @staticmethod
    def _metric_value(metrics: SplitMetrics, name: str) -> float:
        value = getattr(metrics, name)
        return -math.inf if value is None else float(value)

    @staticmethod
    def _finite_or_none(value: Any) -> float | None:
        value = float(value)
        return value if math.isfinite(value) else None

    @staticmethod
    def _parameters(model) -> dict[str, Any]:
        params = model.get_params()
        return {key: params[key] for key in sorted(params)}


def benchmark_from_feature_datasets(
    *,
    train_features: Any,
    validation_features: Any,
    test_features: Any,
    y_train: Sequence[Any],
    y_validation: Sequence[Any],
    y_test: Sequence[Any],
    target: str = "impact_label",
    selection_metric: str = "f1",
) -> BenchmarkReport:
    """Convenience adapter for Phase 10 FeatureDataset objects."""
    feature_names = tuple(train_features.feature_names)
    x_train = [record.values for record in train_features.records]
    x_validation = [record.values for record in validation_features.records]
    x_test = [record.values for record in test_features.records]
    return MLBenchmark().run(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        feature_names=feature_names,
        target=target,
        selection_metric=selection_metric,
    )
