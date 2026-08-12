"""Phase 12 — controlled model training and selection.

Uses an existing Phase 11 benchmark winner as the candidate model family.
Hyperparameter search is small, deterministic, and performed on the training
partition with validation-only selection. The test partition is evaluated
once for the selected configuration.

No feature engineering, random resplitting, or future-outcome feature access
is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Mapping, Sequence

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, average_precision_score,
    roc_auc_score,
)


class TrainingStatus(str, Enum):
    READY = "READY"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class Evaluation:
    sample_size: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    labels: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]

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
            "labels": list(self.labels),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }


@dataclass(frozen=True)
class TrainingCandidate:
    candidate_id: str
    parameters: Mapping[str, Any]
    validation: Evaluation

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parameters": dict(self.parameters),
            "validation": self.validation.to_dict(),
        }


@dataclass(frozen=True)
class ModelArtifact:
    model_family: str
    parameters: Mapping[str, Any]
    feature_names: tuple[str, ...]
    target: str
    selection_metric: str
    threshold: float | None
    model: Any

    def metadata(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "parameters": dict(self.parameters),
            "feature_names": list(self.feature_names),
            "target": self.target,
            "selection_metric": self.selection_metric,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class TrainingReport:
    status: TrainingStatus
    model_family: str
    selected_candidate: str | None
    candidates: tuple[TrainingCandidate, ...]
    validation: Evaluation | None
    test: Evaluation | None
    artifact_metadata: Mapping[str, Any] | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "model_family": self.model_family,
            "selected_candidate": self.selected_candidate,
            "candidates": [c.to_dict() for c in self.candidates],
            "validation": self.validation.to_dict() if self.validation else None,
            "test": self.test.to_dict() if self.test else None,
            "artifact_metadata": dict(self.artifact_metadata) if self.artifact_metadata else None,
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )


class ModelTrainer:
    """Controlled validation-based tuning for a Phase 11 model family."""

    DEFAULT_GRIDS = {
        "logistic_regression": (
            {"C": 0.1, "max_iter": 1000, "class_weight": None},
            {"C": 1.0, "max_iter": 1000, "class_weight": None},
            {"C": 10.0, "max_iter": 1000, "class_weight": None},
            {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
        ),
        "random_forest": (
            {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 1, "class_weight": "balanced"},
            {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 1, "class_weight": "balanced"},
            {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2, "class_weight": "balanced"},
        ),
        "gradient_boosting": (
            {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 2},
            {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
            {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 2},
        ),
        "dummy": (
            {"strategy": "prior"},
        ),
    }

    def train(
        self,
        *,
        model_family: str,
        x_train: Sequence[Sequence[float]],
        y_train: Sequence[Any],
        x_validation: Sequence[Sequence[float]],
        y_validation: Sequence[Any],
        x_test: Sequence[Sequence[float]],
        y_test: Sequence[Any],
        feature_names: Sequence[str],
        target: str = "impact_label",
        selection_metric: str = "f1",
        parameter_grid: Sequence[Mapping[str, Any]] | None = None,
        threshold_candidates: Sequence[float] | None = None,
    ) -> TrainingReport:
        try:
            self._validate(
                model_family, x_train, y_train, x_validation, y_validation,
                x_test, y_test, feature_names, selection_metric,
            )
        except ValueError as exc:
            return TrainingReport(
                TrainingStatus.INVALID_INPUT, model_family, None, (), None, None, None, str(exc)
            )

        grid = tuple(parameter_grid or self.DEFAULT_GRIDS[model_family])
        candidates: list[TrainingCandidate] = []

        for index, params in enumerate(grid):
            model = self._make_model(model_family, params)
            model.fit(x_train, y_train)
            evaluation = self._evaluate(model, x_validation, y_validation)
            candidates.append(
                TrainingCandidate(
                    candidate_id=f"{model_family}-{index + 1}",
                    parameters=dict(params),
                    validation=evaluation,
                )
            )

        selected = max(
            candidates,
            key=lambda candidate: (
                self._metric(candidate.validation, selection_metric),
                -candidates.index(candidate),
            ),
        )

        selected_model = self._make_model(model_family, selected.parameters)
        selected_model.fit(x_train, y_train)
        validation = self._evaluate(selected_model, x_validation, y_validation)

        threshold = None
        if threshold_candidates and self._is_binary(selected_model, y_validation):
            threshold = self._select_threshold(
                selected_model, x_validation, y_validation,
                threshold_candidates, selection_metric,
            )

        # Test is evaluated only after candidate/threshold selection.
        test = self._evaluate(
            selected_model, x_test, y_test, threshold=threshold
        )

        artifact = ModelArtifact(
            model_family=model_family,
            parameters=selected.parameters,
            feature_names=tuple(feature_names),
            target=target,
            selection_metric=selection_metric,
            threshold=threshold,
            model=selected_model,
        )

        return TrainingReport(
            TrainingStatus.READY,
            model_family,
            selected.candidate_id,
            tuple(candidates),
            validation,
            test,
            artifact.metadata(),
        )

    @classmethod
    def _make_model(cls, family: str, params: Mapping[str, Any]):
        params = dict(params)
        if family == "logistic_regression":
            params.setdefault("random_state", 42)
            return LogisticRegression(**params)
        if family == "random_forest":
            params.setdefault("random_state", 42)
            params.setdefault("n_jobs", 1)
            return RandomForestClassifier(**params)
        if family == "gradient_boosting":
            params.setdefault("random_state", 42)
            return GradientBoostingClassifier(**params)
        if family == "dummy":
            return DummyClassifier(**params)
        raise ValueError(f"unsupported model family: {family}")

    @staticmethod
    def _validate(
        family, x_train, y_train, x_validation, y_validation,
        x_test, y_test, feature_names, selection_metric,
    ):
        if family not in ModelTrainer.DEFAULT_GRIDS:
            raise ValueError(f"unsupported model family: {family}")
        if selection_metric not in {"f1", "balanced_accuracy", "pr_auc", "roc_auc"}:
            raise ValueError("unsupported selection metric")
        if not y_train or not y_validation or not y_test:
            raise ValueError("all splits must be non-empty")
        if len(x_train) != len(y_train) or len(x_validation) != len(y_validation) or len(x_test) != len(y_test):
            raise ValueError("X/y lengths differ")
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        width = len(feature_names)
        for matrix in (x_train, x_validation, x_test):
            if any(len(row) != width for row in matrix):
                raise ValueError("feature matrix width differs from feature_names")

    @classmethod
    def _evaluate(cls, model, x, y, threshold=None) -> Evaluation:
        predicted = model.predict(x)
        if threshold is not None and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x)
            predicted = [
                model.classes_[1] if score >= threshold else model.classes_[0]
                for score in probabilities[:, 1]
            ]

        labels = sorted({*map(str, y), *map(str, predicted)})
        matrix = confusion_matrix(
            [str(v) for v in y], [str(v) for v in predicted], labels=labels
        )
        roc_auc = pr_auc = None
        if hasattr(model, "predict_proba") and len(set(y)) == 2:
            try:
                probabilities = model.predict_proba(x)[:, 1]
                positive_class = model.classes_[1]
                positive = [1 if value == positive_class else 0 for value in y]
                roc_auc = float(roc_auc_score(positive, probabilities))
                pr_auc = float(average_precision_score(positive, probabilities))
            except ValueError:
                pass

        return Evaluation(
            len(y),
            float(accuracy_score(y, predicted)),
            float(balanced_accuracy_score(y, predicted)),
            float(precision_score(y, predicted, average="weighted", zero_division=0)),
            float(recall_score(y, predicted, average="weighted", zero_division=0)),
            float(f1_score(y, predicted, average="weighted", zero_division=0)),
            roc_auc,
            pr_auc,
            tuple(labels),
            tuple(tuple(int(v) for v in row) for row in matrix),
        )

    @staticmethod
    def _metric(evaluation: Evaluation, name: str) -> float:
        value = getattr(evaluation, name)
        return -math.inf if value is None else value

    @staticmethod
    def _is_binary(model, y) -> bool:
        return hasattr(model, "classes_") and len(model.classes_) == 2 and len(set(y)) == 2

    @classmethod
    def _select_threshold(cls, model, x, y, thresholds, metric):
        best = None
        for threshold in thresholds:
            if not 0 < threshold < 1:
                raise ValueError("threshold candidates must be between 0 and 1")
            evaluation = cls._evaluate(model, x, y, threshold=threshold)
            score = cls._metric(evaluation, metric)
            candidate = (score, -abs(threshold - 0.5), threshold)
            if best is None or candidate > best:
                best = candidate
        return float(best[2])

    @staticmethod
    def save_metadata(report: TrainingReport, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                report.to_dict(), handle, ensure_ascii=False,
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
