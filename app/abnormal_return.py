from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .expected_return import MarketModel, MarketModelEstimator
from .historical_return_data import AlignedReturn, EventStudyStatus


@dataclass(frozen=True)
class AbnormalReturn:
    offset: int
    trading_date: object
    actual_return: float
    market_return: float
    expected_return: float
    abnormal_return: float
    prediction_variance: float


class AbnormalReturnEngine:
    def calculate(self, model: MarketModel, record: AlignedReturn) -> Optional[AbnormalReturn]:
        expected = MarketModelEstimator.expected(model, record.market_return)
        if expected is None:
            return None
        variance = model.residual_variance * (1 + 1 / model.observations + ((record.market_return - model.market_mean) ** 2 / model.market_sum_squares))
        return AbnormalReturn(record.offset, record.trading_date, record.stock_return,
                              record.market_return, expected, record.stock_return - expected, variance)

    @staticmethod
    def car(items: Sequence[AbnormalReturn]) -> float:
        return sum(item.abnormal_return for item in items)
