"""Phase 19 - forward return prediction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PredictionStatus(str, Enum):
    READY = "READY"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class ReturnPrediction:
    ticker: str
    predicted_return: float
    realized_return: float


@dataclass(frozen=True)
class PredictionMetrics:
    mae: float
    rmse: float
    directional_accuracy: float


@dataclass(frozen=True)
class PredictionReport:
    status: PredictionStatus
    horizon: str
    model_name: str
    metrics: PredictionMetrics | None
    predictions: tuple[ReturnPrediction, ...]
    error: str | None = None


class ForwardReturnPredictor:
    """Deterministic forward-return regression model."""

    SUPPORTED_HORIZONS = ("t1", "t3", "t5", "t10")

    def train_and_evaluate(
        self,
        x_train: Sequence[Sequence[float]],
        y_train: Sequence[float],
        x_test: Sequence[Sequence[float]],
        y_test: Sequence[float],
        tickers: Sequence[str],
        horizon: str = "t5",
    ) -> PredictionReport:
        try:
            self._validate(
                x_train,
                y_train,
                x_test,
                y_test,
                tickers,
                horizon,
            )

            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge(alpha=1.0)),
                ]
            )

            model.fit(x_train, y_train)
            predicted = model.predict(x_test)

            predictions = tuple(
                ReturnPrediction(
                    ticker=ticker,
                    predicted_return=float(pred),
                    realized_return=float(actual),
                )
                for ticker, pred, actual in zip(
                    tickers,
                    predicted,
                    y_test,
                )
            )

            metrics = PredictionMetrics(
                mae=float(mean_absolute_error(y_test, predicted)),
                rmse=float(
                    mean_squared_error(y_test, predicted) ** 0.5
                ),
                directional_accuracy=self._directional_accuracy(
                    y_test,
                    predicted,
                ),
            )

            return PredictionReport(
                status=PredictionStatus.READY,
                horizon=horizon,
                model_name="ridge",
                metrics=metrics,
                predictions=predictions,
            )

        except (ValueError, TypeError) as exc:
            return PredictionReport(
                status=PredictionStatus.INVALID_INPUT,
                horizon=horizon,
                model_name="ridge",
                metrics=None,
                predictions=(),
                error=str(exc),
            )

    @staticmethod
    def _directional_accuracy(actual, predicted) -> float:
        if not actual:
            return 0.0

        correct = sum(
            (float(a) >= 0) == (float(p) >= 0)
            for a, p in zip(actual, predicted)
        )

        return correct / len(actual)

    @classmethod
    def _validate(
        cls,
        x_train,
        y_train,
        x_test,
        y_test,
        tickers,
        horizon,
    ) -> None:
        if horizon not in cls.SUPPORTED_HORIZONS:
            raise ValueError("unsupported horizon")

        if not x_train or not x_test:
            raise ValueError(
                "train and test data must not be empty"
            )

        if len(x_train) != len(y_train):
            raise ValueError(
                "x_train/y_train length mismatch"
            )

        if len(x_test) != len(y_test):
            raise ValueError(
                "x_test/y_test length mismatch"
            )

        if len(x_test) != len(tickers):
            raise ValueError(
                "x_test/tickers length mismatch"
            )

        feature_count = len(x_train[0])

        if feature_count == 0:
            raise ValueError("features must not be empty")

        for row in list(x_train) + list(x_test):
            if len(row) != feature_count:
                raise ValueError(
                    "inconsistent feature dimensions"
                )

            if not all(
                math.isfinite(float(value))
                for value in row
            ):
                raise ValueError(
                    "features must contain finite values"
                )

        for value in list(y_train) + list(y_test):
            if not math.isfinite(float(value)):
                raise ValueError(
                    "targets must contain finite values"
                )

        if any(not ticker for ticker in tickers):
            raise ValueError("ticker must not be empty")
