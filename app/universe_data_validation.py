"""Point-in-time data sufficiency checks for a stock universe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.stock_universe import StockUniverse


@dataclass(frozen=True)
class UniverseDataStatus:
    ticker: str
    eligible: bool
    observation_count: int
    reason: str


class UniverseDataValidator:
    def validate(
        self,
        universe: StockUniverse,
        historical_returns: dict[str, Sequence[float]],
        minimum_observations: int = 60,
    ) -> tuple[UniverseDataStatus, ...]:
        if minimum_observations <= 0:
            raise ValueError("minimum_observations must be positive")

        results = []
        for ticker in universe.tickers:
            values = historical_returns.get(ticker)

            if values is None:
                results.append(
                    UniverseDataStatus(
                        ticker=ticker,
                        eligible=False,
                        observation_count=0,
                        reason="missing historical returns",
                    )
                )
                continue

            count = len(values)
            eligible = count >= minimum_observations
            results.append(
                UniverseDataStatus(
                    ticker=ticker,
                    eligible=eligible,
                    observation_count=count,
                    reason=(
                        "eligible"
                        if eligible
                        else "insufficient historical observations"
                    ),
                )
            )

        return tuple(results)
