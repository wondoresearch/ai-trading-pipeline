"""Phase 20 — transaction cost, slippage and execution modeling.

Evaluation-only layer. Converts gross portfolio period returns and turnover
into net returns using explicit, predeclared execution assumptions.
No calibration or optimization is performed from OOS outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Sequence


class ExecutionStatus(str, Enum):
    READY = "READY"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class ExecutionObservation:
    timestamp: float
    gross_return: float
    turnover: float
    active_exposure: float = 0.0


@dataclass(frozen=True)
class ExecutionConfig:
    commission_rate: float = 0.0
    slippage_rate: float = 0.0
    spread_rate: float = 0.0
    execution_delay_cost: float = 0.0


@dataclass(frozen=True)
class ExecutionMetrics:
    sample_size: int
    gross_total_return: float
    net_total_return: float
    total_execution_cost: float
    average_cost_per_period: float
    cost_drag: float
    gross_maximum_drawdown: float
    net_maximum_drawdown: float
    gross_sharpe_like: float | None
    net_sharpe_like: float | None
    break_even_cost_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "gross_total_return": self.gross_total_return,
            "net_total_return": self.net_total_return,
            "total_execution_cost": self.total_execution_cost,
            "average_cost_per_period": self.average_cost_per_period,
            "cost_drag": self.cost_drag,
            "gross_maximum_drawdown": self.gross_maximum_drawdown,
            "net_maximum_drawdown": self.net_maximum_drawdown,
            "gross_sharpe_like": self.gross_sharpe_like,
            "net_sharpe_like": self.net_sharpe_like,
            "break_even_cost_rate": self.break_even_cost_rate,
        }


@dataclass(frozen=True)
class ExecutionReport:
    status: ExecutionStatus
    metrics: ExecutionMetrics | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )


class ExecutionModel:
    """Apply deterministic execution friction to frozen gross returns."""

    def evaluate(
        self,
        observations: Sequence[ExecutionObservation],
        config: ExecutionConfig,
    ) -> ExecutionReport:
        try:
            obs = tuple(observations)
            self._validate(obs, config)

            gross = [float(x.gross_return) for x in obs]
            costs = [
                abs(float(x.turnover)) * (
                    config.commission_rate
                    + config.slippage_rate
                    + config.spread_rate
                )
                + abs(float(x.active_exposure)) * config.execution_delay_cost
                for x in obs
            ]
            net = [r - c for r, c in zip(gross, costs)]

            gross_total = math.prod(1 + r for r in gross) - 1
            net_total = math.prod(1 + r for r in net) - 1

            gross_dd = self._max_drawdown(gross)
            net_dd = self._max_drawdown(net)
            gross_sharpe = self._sharpe(gross)
            net_sharpe = self._sharpe(net)

            avg_cost = sum(costs) / len(costs)
            total_cost = sum(costs)
            cost_drag = net_total - gross_total

            # Approximate break-even proportional execution rate using
            # observed turnover. This is a descriptive sensitivity threshold,
            # not a parameter fit.
            total_turnover = sum(abs(x.turnover) for x in obs)
            break_even = (
                gross_total / total_turnover
                if total_turnover > 0 and gross_total > 0
                else None
            )

            return ExecutionReport(
                ExecutionStatus.READY,
                ExecutionMetrics(
                    len(obs), gross_total, net_total, total_cost,
                    avg_cost, cost_drag, gross_dd, net_dd,
                    gross_sharpe, net_sharpe, break_even,
                ),
            )
        except (ValueError, TypeError, OverflowError, IndexError) as exc:
            return ExecutionReport(ExecutionStatus.INVALID_INPUT, None, str(exc))

    @staticmethod
    def _max_drawdown(returns: Sequence[float]) -> float:
        equity = peak = 1.0
        drawdown = 0.0
        for r in returns:
            equity *= 1 + r
            peak = max(peak, equity)
            drawdown = min(drawdown, equity / peak - 1)
        return drawdown

    @staticmethod
    def _sharpe(returns: Sequence[float]) -> float | None:
        if len(returns) < 2:
            return None
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        return None if std == 0 else mean / std * math.sqrt(len(returns))

    @staticmethod
    def _validate(
        observations: Sequence[ExecutionObservation],
        config: ExecutionConfig,
    ) -> None:
        if not observations:
            raise ValueError("observations must not be empty")
        if any(
            not math.isfinite(float(x.gross_return))
            or not math.isfinite(float(x.turnover))
            or not math.isfinite(float(x.active_exposure))
            for x in observations
        ):
            raise ValueError("observation values must be finite")
        timestamps = [x.timestamp for x in observations]
        if timestamps != sorted(timestamps):
            raise ValueError("observations must be chronologically sorted")
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("timestamps must be unique")
        for value in (
            config.commission_rate,
            config.slippage_rate,
            config.spread_rate,
            config.execution_delay_cost,
        ):
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError("execution rates must be finite and non-negative")
