"""End-to-end research opportunity ranking orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.opportunity_ranking import OpportunityRanker, StockOpportunity
from app.stock_universe import StockUniverse
from app.universe_data_validation import (
    UniverseDataStatus,
    UniverseDataValidator,
)


@dataclass(frozen=True)
class OpportunityPipelineResult:
    universe: StockUniverse
    data_status: tuple[UniverseDataStatus, ...]
    ranking: tuple[StockOpportunity, ...]


class OpportunityPipeline:
    def __init__(
        self,
        validator: UniverseDataValidator | None = None,
        ranker: OpportunityRanker | None = None,
    ) -> None:
        self.validator = validator or UniverseDataValidator()
        self.ranker = ranker or OpportunityRanker()

    def run(
        self,
        tickers: Sequence[str],
        predictions: dict[str, float],
        confidence: dict[str, float],
        historical_returns: dict[str, Sequence[float]],
        market_returns: Sequence[float] | None = None,
        data_minimum_observations: int = 60,
        risk_minimum_observations: int = 20,
    ) -> OpportunityPipelineResult:
        universe = StockUniverse.from_tickers(tickers)

        status = self.validator.validate(
            universe,
            historical_returns,
            minimum_observations=data_minimum_observations,
        )

        eligible = {
            item.ticker
            for item in status
            if item.eligible
        }

        if not eligible:
            return OpportunityPipelineResult(
                universe=universe,
                data_status=status,
                ranking=(),
            )

        eligible_universe = StockUniverse.from_tickers(
            ticker for ticker in universe.tickers if ticker in eligible
        )

        ranking = self.ranker.rank(
            universe=eligible_universe,
            predictions=predictions,
            confidence=confidence,
            historical_returns=historical_returns,
            market_returns=market_returns,
            minimum_observations=risk_minimum_observations,
        )

        return OpportunityPipelineResult(
            universe=universe,
            data_status=status,
            ranking=ranking,
        )
