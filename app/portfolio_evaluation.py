"""Phase 19 — portfolio-level evaluation and risk aggregation.

Evaluation-only layer over frozen per-ticker OOS returns and signals.
No future return is used for position sizing, and portfolio weights are
supplied explicitly rather than optimized from OOS outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Sequence


class PortfolioStatus(str, Enum):
    READY = "READY"
    INVALID_INPUT = "INVALID_INPUT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class AssetObservation:
    timestamp: float
    ticker: str
    strategy_return: float
    signal: int = 0


@dataclass(frozen=True)
class PortfolioConfig:
    weights: dict[str, float]
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    max_ticker_weight: float = 1.0


@dataclass(frozen=True)
class PortfolioMetrics:
    sample_size: int
    total_return: float
    mean_return: float
    volatility: float
    maximum_drawdown: float
    sharpe_like: float | None
    average_gross_exposure: float
    average_net_exposure: float
    turnover: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "total_return": self.total_return,
            "mean_return": self.mean_return,
            "volatility": self.volatility,
            "maximum_drawdown": self.maximum_drawdown,
            "sharpe_like": self.sharpe_like,
            "average_gross_exposure": self.average_gross_exposure,
            "average_net_exposure": self.average_net_exposure,
            "turnover": self.turnover,
        }


@dataclass(frozen=True)
class PortfolioAttribution:
    ticker: str
    contribution: float

    def to_dict(self) -> dict[str, Any]:
        return {"ticker": self.ticker, "contribution": self.contribution}


@dataclass(frozen=True)
class PortfolioReport:
    status: PortfolioStatus
    metrics: PortfolioMetrics | None
    attribution: tuple[PortfolioAttribution, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "attribution": [x.to_dict() for x in self.attribution],
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class PortfolioEvaluator:
    """Evaluate explicitly weighted assets on a common chronological timeline."""

    def evaluate(
        self,
        observations: Sequence[AssetObservation],
        config: PortfolioConfig,
    ) -> PortfolioReport:
        try:
            obs = tuple(observations)
            self._validate(obs, config)

            by_time: dict[float, list[AssetObservation]] = {}
            for item in obs:
                by_time.setdefault(item.timestamp, []).append(item)

            timestamps = sorted(by_time)
            portfolio_returns: list[float] = []
            gross: list[float] = []
            net: list[float] = []
            attribution_totals = {ticker: 0.0 for ticker in config.weights}
            previous_weights: dict[str, float] = {}

            for timestamp in timestamps:
                rows = by_time[timestamp]
                row_map = {x.ticker: x for x in rows}
                active_weights: dict[str, float] = {}

                for ticker, base_weight in config.weights.items():
                    item = row_map.get(ticker)
                    signal = 0 if item is None else item.signal
                    active_weights[ticker] = base_weight * signal

                gross_exp = sum(abs(x) for x in active_weights.values())
                net_exp = sum(active_weights.values())

                if gross_exp > config.max_gross_exposure + 1e-12:
                    raise ValueError(
                        f"gross exposure exceeds limit at timestamp {timestamp}"
                    )
                if abs(net_exp) > config.max_net_exposure + 1e-12:
                    raise ValueError(
                        f"net exposure exceeds limit at timestamp {timestamp}"
                    )

                period_return = 0.0
                for ticker, weight in active_weights.items():
                    item = row_map.get(ticker)
                    contribution = 0.0 if item is None else weight * item.strategy_return
                    period_return += contribution
                    attribution_totals[ticker] += contribution

                portfolio_returns.append(period_return)
                gross.append(gross_exp)
                net.append(net_exp)

                if previous_weights:
                    # One-way turnover on changes in signed target weights.
                    turnover = sum(
                        abs(active_weights.get(t, 0.0) - previous_weights.get(t, 0.0))
                        for t in set(active_weights) | set(previous_weights)
                    )
                else:
                    turnover = sum(abs(x) for x in active_weights.values())
                previous_weights = active_weights
                if len(portfolio_returns) == 1:
                    turnover_total = turnover
                else:
                    turnover_total += turnover

            metrics = self._metrics(
                portfolio_returns, gross, net, turnover_total
            )
            attribution = tuple(
                PortfolioAttribution(ticker, attribution_totals[ticker])
                for ticker in sorted(attribution_totals)
            )
            return PortfolioReport(
                PortfolioStatus.READY, metrics, attribution
            )
        except (ValueError, TypeError, IndexError, OverflowError) as exc:
            return PortfolioReport(
                PortfolioStatus.INVALID_INPUT, None, (), str(exc)
            )

    @staticmethod
    def _metrics(
        returns: Sequence[float],
        gross: Sequence[float],
        net: Sequence[float],
        turnover: float,
    ) -> PortfolioMetrics:
        total = math.prod(1.0 + x for x in returns) - 1.0
        mean = sum(returns) / len(returns)
        variance = (
            sum((x - mean) ** 2 for x in returns) / (len(returns) - 1)
            if len(returns) > 1 else 0.0
        )
        volatility = math.sqrt(variance)

        equity = peak = 1.0
        max_dd = 0.0
        for value in returns:
            equity *= 1.0 + value
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / peak - 1.0)

        sharpe = (
            mean / volatility * math.sqrt(len(returns))
            if len(returns) > 1 and volatility > 0 else None
        )
        return PortfolioMetrics(
            len(returns), total, mean, volatility, max_dd, sharpe,
            sum(gross) / len(gross), sum(net) / len(net), turnover
        )

    @staticmethod
    def _validate(
        observations: Sequence[AssetObservation],
        config: PortfolioConfig,
    ) -> None:
        if not observations:
            raise ValueError("observations must not be empty")
        if not config.weights:
            raise ValueError("weights must not be empty")
        if config.max_gross_exposure < 0:
            raise ValueError("max_gross_exposure must be non-negative")
        if config.max_net_exposure < 0:
            raise ValueError("max_net_exposure must be non-negative")
        if config.max_ticker_weight < 0:
            raise ValueError("max_ticker_weight must be non-negative")

        for ticker, weight in config.weights.items():
            if not ticker:
                raise ValueError("ticker must not be empty")
            if not math.isfinite(float(weight)):
                raise ValueError("weights must be finite")
            if abs(weight) > config.max_ticker_weight + 1e-12:
                raise ValueError(f"ticker weight exceeds limit: {ticker}")

        timestamps = [x.timestamp for x in observations]
        if timestamps != sorted(timestamps):
            raise ValueError("observations must be chronologically sorted")

        for item in observations:
            if item.ticker not in config.weights:
                raise ValueError(f"unknown ticker: {item.ticker}")
            if not math.isfinite(float(item.strategy_return)):
                raise ValueError("strategy_return must be finite")
            if item.signal not in (-1, 0, 1):
                raise ValueError("signal must be -1, 0, or 1")
