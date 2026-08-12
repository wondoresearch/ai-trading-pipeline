"""Phase 13 — blind out-of-sample prediction evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence
import json

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
)


class OOSStatus(str, Enum):
    READY = "READY"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class OOSPrediction:
    event_id: str
    ticker: str
    actual_label: Any
    predicted_label: Any
    probability_positive: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "actual_label": self.actual_label,
            "predicted_label": self.predicted_label,
            "probability_positive": self.probability_positive,
        }


@dataclass(frozen=True)
class OOSEvaluation:
    status: OOSStatus
    sample_size: int
    accuracy: float | None
    balanced_accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    roc_auc: float | None
    pr_auc: float | None
    predictions: tuple[OOSPrediction, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "sample_size": self.sample_size,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "predictions": [p.to_dict() for p in self.predictions],
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)


class OOSEvaluator:
    """Evaluate an already-fitted model without fitting or tuning it."""

    def evaluate(
        self,
        *,
        model: Any,
        x_oos: Sequence[Sequence[float]],
        y_oos: Sequence[Any],
        event_ids: Sequence[str],
        tickers: Sequence[str],
        threshold: float | None = None,
    ) -> OOSEvaluation:
        try:
            if not y_oos:
                raise ValueError("OOS dataset must not be empty")
            if len(x_oos) != len(y_oos) or len(y_oos) != len(event_ids) or len(y_oos) != len(tickers):
                raise ValueError("OOS input lengths differ")
            if threshold is not None and not 0 < threshold < 1:
                raise ValueError("threshold must be between 0 and 1")
            if not hasattr(model, "predict"):
                raise ValueError("model must provide predict()")
            predicted = list(model.predict(x_oos))
            probabilities = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_oos)
                if getattr(proba, "shape", (0, 0))[1] == 2:
                    probabilities = [float(v) for v in proba[:, 1]]
                    if threshold is not None:
                        predicted = [
                            model.classes_[1] if p >= threshold else model.classes_[0]
                            for p in probabilities
                        ]

            labels = sorted({*map(str, y_oos), *map(str, predicted)})
            matrix = confusion_matrix(
                [str(v) for v in y_oos], [str(v) for v in predicted], labels=labels
            )
            _ = matrix  # matrix is deliberately computed for validation of label space.

            roc_auc = pr_auc = None
            if probabilities is not None and len(set(y_oos)) == 2:
                positive_class = model.classes_[1]
                binary = [1 if value == positive_class else 0 for value in y_oos]
                try:
                    roc_auc = float(roc_auc_score(binary, probabilities))
                    pr_auc = float(average_precision_score(binary, probabilities))
                except ValueError:
                    pass

            predictions = tuple(
                OOSPrediction(str(event_ids[i]), str(tickers[i]), y_oos[i],
                              predicted[i], probabilities[i] if probabilities else None)
                for i in range(len(y_oos))
            )
            return OOSEvaluation(
                OOSStatus.READY, len(y_oos),
                float(accuracy_score(y_oos, predicted)),
                float(balanced_accuracy_score(y_oos, predicted)),
                float(precision_score(y_oos, predicted, average="weighted", zero_division=0)),
                float(recall_score(y_oos, predicted, average="weighted", zero_division=0)),
                float(f1_score(y_oos, predicted, average="weighted", zero_division=0)),
                roc_auc, pr_auc, predictions,
            )
        except (ValueError, TypeError, AttributeError) as exc:
            return OOSEvaluation(OOSStatus.INVALID_INPUT, 0, None, None, None, None, None,
                                  None, None, (), str(exc))
