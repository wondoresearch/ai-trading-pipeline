"""Rank a user-defined stock universe by risk-adjusted opportunity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.opportunity_score import OpportunityScore, OpportunityScorer
from app.risk_model import RiskEstimator, RiskMetrics
from app.stock_universe import StockUniverse


@dataclass(frozen=True)
class StockOpportunity:
    ticker: str
    prediction: float
    confidence: float
    risk: RiskMetrics
    opportunity: OpportunityScore
    rank: int


class OpportunityRanker:
    def __init__(
        self,
        risk_estimator: RiskEstimator | None = None,
        scorer: OpportunityScorer | None = None,
    ) -> None:
        self.risk_estimator = risk_estimator or RiskEstimator()
        self.scorer = scorer or OpportunityScorer()

    def rank(
        self,
        universe: StockUniverse,
        predictions: dict[str, float],
        confidence: dict[str, float],
        historical_returns: dict[str, Sequence[float]],
        market_returns: Sequence[float] | None = None,
        minimum_observations: int = 20,
    ) -> tuple[StockOpportunity, ...]:
        rows = []

        for ticker in universe.tickers:
            if ticker not in predictions:
                raise ValueError(f"missing prediction for {ticker}")
            if ticker not in confidence:
                raise ValueError(f"missing confidence for {ticker}")
            if ticker not in historical_returns:
                raise ValueError(f"missing historical returns for {ticker}")

            risk = self.risk_estimator.estimate(
                historical_returns[ticker],
                market_returns=market_returns,
                minimum_observations=minimum_observations,
            )
            opportunity = self.scorer.score(
                expected_return=predictions[ticker],
                confidence=confidence[ticker],
                volatility=risk.volatility,
                downside_deviation=risk.downside_deviation,
                max_drawdown=risk.max_drawdown,
                beta=risk.beta,
            )
            rows.append(
                (
                    ticker,
                    predictions[ticker],
                    confidence[ticker],
                    risk,
                    opportunity,
                )
            )

        rows.sort(
            key=lambda row: (
                row[4].score,
                row[4].expected_return,
                row[1],
            ),
            reverse=True,
        )

        return tuple(
            StockOpportunity(
                ticker=row[0],
                prediction=float(row[1]),
                confidence=float(row[2]),
                risk=row[3],
                opportunity=row[4],
                rank=index,
            )
            for index, row in enumerate(rows, start=1)
        )
