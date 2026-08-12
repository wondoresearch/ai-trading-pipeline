"""Phase 14 — OOS strategy benchmark and robustness comparison."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence
import json
import math


class BenchmarkStatus(str, Enum):
    READY = "READY"
    INVALID_INPUT = "INVALID_INPUT"


class BenchmarkStrategy(str, Enum):
    MODEL = "MODEL"
    ALWAYS_LONG = "ALWAYS_LONG"
    ALWAYS_SHORT = "ALWAYS_SHORT"
    NO_POSITION = "NO_POSITION"


@dataclass(frozen=True)
class BenchmarkObservation:
    event_id: str
    ticker: str
    model_signal: str
    realized_return: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "model_signal": self.model_signal,
            "realized_return": self.realized_return,
        }


@dataclass(frozen=True)
class StrategyMetrics:
    strategy: BenchmarkStrategy
    observations: int
    trades: int
    total_return: float | None
    average_return: float | None
    win_rate: float | None
    volatility: float | None
    sharpe_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "observations": self.observations,
            "trades": self.trades,
            "total_return": self.total_return,
            "average_return": self.average_return,
            "win_rate": self.win_rate,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    status: BenchmarkStatus
    observations: int
    strategies: tuple[StrategyMetrics, ...]
    model_minus_always_long: float | None
    model_minus_always_short: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "observations": self.observations,
            "strategies": [s.to_dict() for s in self.strategies],
            "model_minus_always_long": self.model_minus_always_long,
            "model_minus_always_short": self.model_minus_always_short,
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )


class StrategyBenchmarkEngine:
    """Compare a frozen OOS model signal with deterministic baselines.

    This component never trains, tunes, or selects a model. All strategies
    consume exactly the same realized event-return observations.
    """

    def compare(
        self,
        observations: Sequence[BenchmarkObservation],
        *,
        annualization_periods: int = 252,
        strategies: Sequence[BenchmarkStrategy] = (
            BenchmarkStrategy.MODEL,
            BenchmarkStrategy.ALWAYS_LONG,
            BenchmarkStrategy.ALWAYS_SHORT,
            BenchmarkStrategy.NO_POSITION,
        ),
    ) -> BenchmarkReport:
        try:
            if annualization_periods <= 0:
                raise ValueError("annualization_periods must be positive")
            if not observations:
                raise ValueError("observations must not be empty")
            if len(set(o.event_id for o in observations)) != len(observations):
                raise ValueError("event_id must be unique")
            selected = tuple(dict.fromkeys(strategies))
            if not selected:
                raise ValueError("strategies must not be empty")

            metrics = tuple(
                self._metrics(observations, strategy, annualization_periods)
                for strategy in selected
            )
            lookup = {m.strategy: m for m in metrics}
            model = lookup.get(BenchmarkStrategy.MODEL)
            long = lookup.get(BenchmarkStrategy.ALWAYS_LONG)
            short = lookup.get(BenchmarkStrategy.ALWAYS_SHORT)
            return BenchmarkReport(
                BenchmarkStatus.READY,
                len(observations),
                metrics,
                self._delta(model, long),
                self._delta(model, short),
            )
        except (ValueError, TypeError) as exc:
            return BenchmarkReport(
                BenchmarkStatus.INVALID_INPUT, 0, (), None, None, str(exc)
            )

    @staticmethod
    def _signed_return(observation: BenchmarkObservation, strategy: BenchmarkStrategy) -> float | None:
        value = observation.realized_return
        if value is None:
            return None
        if strategy == BenchmarkStrategy.NO_POSITION:
            return 0.0
        if strategy == BenchmarkStrategy.ALWAYS_LONG:
            return float(value)
        if strategy == BenchmarkStrategy.ALWAYS_SHORT:
            return float(-value)
        if strategy == BenchmarkStrategy.MODEL:
            if observation.model_signal == "LONG":
                return float(value)
            if observation.model_signal == "SHORT":
                return float(-value)
            return 0.0
        raise ValueError(f"unsupported strategy: {strategy}")

    def _metrics(
        self,
        observations: Sequence[BenchmarkObservation],
        strategy: BenchmarkStrategy,
        annualization_periods: int,
    ) -> StrategyMetrics:
        returns = [
            signed for signed in (
                self._signed_return(o, strategy) for o in observations
            )
            if signed is not None
        ]
        trades = sum(
            1 for o in observations
            if strategy in (BenchmarkStrategy.ALWAYS_LONG, BenchmarkStrategy.ALWAYS_SHORT)
            or (strategy == BenchmarkStrategy.MODEL and o.model_signal in ("LONG", "SHORT"))
        )
        if not returns:
            return StrategyMetrics(strategy, 0, trades, None, None, None, None, None)
        equity = math.prod(1.0 + r for r in returns)
        total = equity - 1.0
        average = sum(returns) / len(returns)
        variance = sum((r - average) ** 2 for r in returns) / len(returns)
        volatility = math.sqrt(variance) * math.sqrt(annualization_periods)
        sharpe = (
            average / math.sqrt(variance) * math.sqrt(annualization_periods)
            if variance > 0 else None
        )
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        return StrategyMetrics(
            strategy, len(returns), trades, total, average,
            win_rate, volatility, sharpe,
        )

    @staticmethod
    def _delta(
        model: StrategyMetrics | None,
        baseline: StrategyMetrics | None,
    ) -> float | None:
        if model is None or baseline is None:
            return None
        if model.total_return is None or baseline.total_return is None:
            return None
        return model.total_return - baseline.total_return
