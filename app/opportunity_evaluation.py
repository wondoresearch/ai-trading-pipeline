"""Phase 19 - integrated opportunity evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.cross_sectional_evaluation import (
    CrossSectionalEvaluator,
    CrossSectionalMetrics,
)
from app.forward_return_prediction import (
    ForwardReturnPredictor,
    PredictionReport,
)


@dataclass(frozen=True)
class OpportunityEvaluationReport:
    prediction: PredictionReport
    cross_sectional: CrossSectionalMetrics


class OpportunityEvaluator:
    """Combine forward-return prediction and cross-sectional evaluation."""

    def __init__(
        self,
        predictor: ForwardReturnPredictor | None = None,
        cross_sectional: CrossSectionalEvaluator | None = None,
    ) -> None:
        self.predictor = predictor or ForwardReturnPredictor()
        self.cross_sectional = (
            cross_sectional or CrossSectionalEvaluator()
        )

    def evaluate(
        self,
        x_train: Sequence[Sequence[float]],
        y_train: Sequence[float],
        x_test: Sequence[Sequence[float]],
        y_test: Sequence[float],
        tickers: Sequence[str],
        benchmark_returns: Sequence[float],
        horizon: str = "t5",
        top_k: int = 3,
    ) -> OpportunityEvaluationReport:
        prediction = self.predictor.train_and_evaluate(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            tickers=tickers,
            horizon=horizon,
        )

        if prediction.metrics is None:
            raise ValueError(
                prediction.error or "prediction failed"
            )

        predicted_returns = [
            item.predicted_return
            for item in prediction.predictions
        ]

        cross_sectional = self.cross_sectional.evaluate(
            predicted_returns=predicted_returns,
            realized_returns=y_test,
            benchmark_returns=benchmark_returns,
            top_k=top_k,
        )

        return OpportunityEvaluationReport(
            prediction=prediction,
            cross_sectional=cross_sectional,
        )
