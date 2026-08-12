from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .historical_return_data import AlignedReturn, EventStudyStatus


@dataclass(frozen=True)
class MarketModel:
    alpha: Optional[float]
    beta: Optional[float]
    residual_variance: Optional[float]
    market_mean: Optional[float]
    market_sum_squares: Optional[float]
    observations: int
    status: EventStudyStatus


class MarketModelEstimator:
    MINIMUM_OBSERVATIONS = 120

    def fit(self, returns: Sequence[AlignedReturn]) -> MarketModel:
        if len(returns) < self.MINIMUM_OBSERVATIONS:
            return MarketModel(None, None, None, None, None, len(returns), EventStudyStatus.INSUFFICIENT_ESTIMATION_DATA)
        y = np.array([item.stock_return for item in returns], dtype=float)
        x = np.array([item.market_return for item in returns], dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return MarketModel(None, None, None, None, None, len(returns), EventStudyStatus.INVALID_RETURN_DATA)
        mean = float(x.mean()); sxx = float(((x - mean) ** 2).sum())
        if sxx <= 0:
            return MarketModel(None, None, None, mean, sxx, len(returns), EventStudyStatus.MODEL_FAILURE)
        beta = float(((x - mean) * (y - y.mean())).sum() / sxx)
        alpha = float(y.mean() - beta * mean)
        residuals = y - (alpha + beta * x)
        variance = float((residuals ** 2).sum() / (len(returns) - 2))
        if not np.isfinite(variance):
            return MarketModel(None, None, None, mean, sxx, len(returns), EventStudyStatus.MODEL_FAILURE)
        return MarketModel(alpha, beta, variance, mean, sxx, len(returns), EventStudyStatus.OBSERVED)

    @staticmethod
    def expected(model: MarketModel, market_return: float) -> Optional[float]:
        if model.status is not EventStudyStatus.OBSERVED:
            return None
        return model.alpha + model.beta * market_return
