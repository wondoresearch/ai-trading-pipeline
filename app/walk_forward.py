"""Phase 16 — walk-forward temporal robustness validation.

Consumes precomputed chronological feature/label rows and a caller-supplied
training function. The evaluator enforces temporal boundaries and evaluates
only the immediately following OOS window for each fold.

No shuffling, random K-fold, or OOS-driven selection is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Callable, Sequence


class WalkForwardStatus(str, Enum):
    READY = "READY"
    INVALID_INPUT = "INVALID_INPUT"
    INSUFFICIENT_FOLD_DATA = "INSUFFICIENT_FOLD_DATA"


@dataclass(frozen=True)
class TemporalRow:
    timestamp: float
    features: tuple[float, ...]
    label: str
    realized_return: float | None = None


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_start: float
    train_end: float
    validation_start: float
    validation_end: float
    oos_start: float
    oos_end: float
    train_size: int
    validation_size: int
    oos_size: int
    threshold: float | None
    oos_returns: tuple[float, ...]
    oos_total_return: float | None
    oos_mean_return: float | None
    positive_return_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "oos_start": self.oos_start,
            "oos_end": self.oos_end,
            "train_size": self.train_size,
            "validation_size": self.validation_size,
            "oos_size": self.oos_size,
            "threshold": self.threshold,
            "oos_returns": list(self.oos_returns),
            "oos_total_return": self.oos_total_return,
            "oos_mean_return": self.oos_mean_return,
            "positive_return_ratio": self.positive_return_ratio,
        }


@dataclass(frozen=True)
class WalkForwardReport:
    status: WalkForwardStatus
    folds: tuple[FoldResult, ...]
    aggregate_oos_total_return: float | None
    aggregate_oos_mean_return: float | None
    positive_fold_ratio: float | None
    worst_fold_total_return: float | None
    best_fold_total_return: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "folds": [f.to_dict() for f in self.folds],
            "aggregate_oos_total_return": self.aggregate_oos_total_return,
            "aggregate_oos_mean_return": self.aggregate_oos_mean_return,
            "positive_fold_ratio": self.positive_fold_ratio,
            "worst_fold_total_return": self.worst_fold_total_return,
            "best_fold_total_return": self.best_fold_total_return,
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )


@dataclass(frozen=True)
class FoldPlan:
    train_size: int
    validation_size: int
    oos_size: int
    step_size: int | None = None


@dataclass(frozen=True)
class TrainedFoldModel:
    """Minimal frozen fold artifact required by the evaluator."""
    predict_signal: Callable[[Sequence[float]], str]
    threshold: float | None = None


TrainFn = Callable[
    [Sequence[TemporalRow], Sequence[TemporalRow]],
    TrainedFoldModel,
]


class WalkForwardEvaluator:
    """Execute chronological train -> validation -> OOS folds."""

    def evaluate(
        self,
        rows: Sequence[TemporalRow],
        plan: FoldPlan,
        train_fn: TrainFn,
    ) -> WalkForwardReport:
        try:
            ordered = tuple(rows)
            self._validate_rows(ordered)
            self._validate_plan(plan)
            if not callable(train_fn):
                raise ValueError("train_fn must be callable")

            folds: list[FoldResult] = []
            start = 0
            fold_number = 1
            step = plan.step_size or plan.oos_size

            while start + plan.train_size + plan.validation_size + plan.oos_size <= len(ordered):
                train = ordered[start:start + plan.train_size]
                validation_start = start + plan.train_size
                validation = ordered[
                    validation_start:validation_start + plan.validation_size
                ]
                oos_start = validation_start + plan.validation_size
                oos = ordered[oos_start:oos_start + plan.oos_size]

                if not train or not validation or not oos:
                    break

                model = train_fn(train, validation)
                if not isinstance(model, TrainedFoldModel):
                    raise ValueError("train_fn must return TrainedFoldModel")

                realized: list[float] = []
                for row in oos:
                    signal = model.predict_signal(row.features)
                    if signal not in ("LONG", "SHORT", "NO_POSITION"):
                        raise ValueError("predict_signal returned invalid signal")
                    if row.realized_return is None:
                        continue
                    if signal == "LONG":
                        realized.append(float(row.realized_return))
                    elif signal == "SHORT":
                        realized.append(float(-row.realized_return))
                    else:
                        realized.append(0.0)

                if not realized:
                    raise ValueError(f"fold {fold_number} has no realized OOS returns")

                total = math.prod(1.0 + r for r in realized) - 1.0
                mean = sum(realized) / len(realized)
                positive = sum(r > 0 for r in realized) / len(realized)

                folds.append(
                    FoldResult(
                        fold_number,
                        train[0].timestamp,
                        train[-1].timestamp,
                        validation[0].timestamp,
                        validation[-1].timestamp,
                        oos[0].timestamp,
                        oos[-1].timestamp,
                        len(train),
                        len(validation),
                        len(oos),
                        model.threshold,
                        tuple(realized),
                        total,
                        mean,
                        positive,
                    )
                )
                fold_number += 1
                start += step

            if not folds:
                return WalkForwardReport(
                    WalkForwardStatus.INSUFFICIENT_FOLD_DATA,
                    (), None, None, None, None, None,
                    "no complete walk-forward fold is available",
                )

            all_returns = [
                r for fold in folds for r in fold.oos_returns
            ]
            aggregate_total = math.prod(1.0 + r for r in all_returns) - 1.0
            aggregate_mean = sum(all_returns) / len(all_returns)
            positive_fold_ratio = sum(
                fold.oos_total_return is not None and fold.oos_total_return > 0
                for fold in folds
            ) / len(folds)
            totals = [fold.oos_total_return for fold in folds if fold.oos_total_return is not None]

            return WalkForwardReport(
                WalkForwardStatus.READY,
                tuple(folds),
                aggregate_total,
                aggregate_mean,
                positive_fold_ratio,
                min(totals),
                max(totals),
            )
        except (ValueError, TypeError, IndexError, OverflowError) as exc:
            return WalkForwardReport(
                WalkForwardStatus.INVALID_INPUT,
                (), None, None, None, None, None, str(exc),
            )

    @staticmethod
    def _validate_rows(rows: Sequence[TemporalRow]) -> None:
        if not rows:
            raise ValueError("rows must not be empty")
        timestamps = [row.timestamp for row in rows]
        if timestamps != sorted(timestamps):
            raise ValueError("rows must be chronologically sorted")
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("timestamps must be unique")
        for row in rows:
            if not isinstance(row.features, tuple):
                raise ValueError("features must be tuples")
            if not row.features:
                raise ValueError("features must not be empty")
            if not all(math.isfinite(float(x)) for x in row.features):
                raise ValueError("features must be finite")
            if row.realized_return is not None and not math.isfinite(float(row.realized_return)):
                raise ValueError("realized_return must be finite")

    @staticmethod
    def _validate_plan(plan: FoldPlan) -> None:
        if plan.train_size <= 0 or plan.validation_size <= 0 or plan.oos_size <= 0:
            raise ValueError("all fold sizes must be positive")
        if plan.step_size is not None and plan.step_size <= 0:
            raise ValueError("step_size must be positive")
