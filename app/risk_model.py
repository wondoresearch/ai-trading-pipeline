"""Risk estimation for research-only opportunity ranking.

No future return target is used here. Inputs are historical return series
and optional market return series available at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, pstdev
from typing import Sequence


@dataclass(frozen=True)
class RiskMetrics:
    volatility: float
    downside_deviation: float
    max_drawdown: float
    beta: float | None
    observation_count: int


class RiskEstimator:
    def estimate(
        self,
        returns: Sequence[float],
        market_returns: Sequence[float] | None = None,
        minimum_observations: int = 20,
    ) -> RiskMetrics:
        self._validate(returns, minimum_observations)

        volatility = pstdev(returns) * math.sqrt(252.0)
        negative = [value for value in returns if value < 0]
        downside = (
            pstdev(negative) * math.sqrt(252.0)
            if len(negative) >= 2
            else 0.0
        )
        max_drawdown = self._max_drawdown(returns)

        beta = None
        if market_returns is not None:
            self._validate_market(returns, market_returns)
            beta = self._beta(returns, market_returns)

        return RiskMetrics(
            volatility=float(volatility),
            downside_deviation=float(downside),
            max_drawdown=float(max_drawdown),
            beta=None if beta is None else float(beta),
            observation_count=len(returns),
        )

    @staticmethod
    def _max_drawdown(returns: Sequence[float]) -> float:
        wealth = 1.0
        peak = 1.0
        worst = 0.0
        for value in returns:
            wealth *= 1.0 + value
            peak = max(peak, wealth)
            drawdown = (wealth / peak) - 1.0
            worst = min(worst, drawdown)
        return abs(worst)

    @staticmethod
    def _beta(
        returns: Sequence[float],
        market_returns: Sequence[float],
    ) -> float | None:
        if len(returns) < 2:
            return None
        x_bar = mean(market_returns)
        y_bar = mean(returns)
        covariance = sum(
            (x - x_bar) * (y - y_bar)
            for x, y in zip(market_returns, returns)
        )
        variance = sum((x - x_bar) ** 2 for x in market_returns)
        if variance == 0:
            return None
        return covariance / variance

    @staticmethod
    def _validate(
        returns: Sequence[float],
        minimum_observations: int,
    ) -> None:
        if len(returns) < minimum_observations:
            raise ValueError(
                f"insufficient observations: {len(returns)} < "
                f"{minimum_observations}"
            )
        if not all(math.isfinite(float(value)) for value in returns):
            raise ValueError("returns must contain finite values")

    @staticmethod
    def _validate_market(
        returns: Sequence[float],
        market_returns: Sequence[float],
    ) -> None:
        if len(returns) != len(market_returns):
            raise ValueError("stock/market return length mismatch")
        if not all(
            math.isfinite(float(value)) for value in market_returns
        ):
            raise ValueError("market returns must contain finite values")
