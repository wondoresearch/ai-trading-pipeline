"""Phase 13 — event-driven out-of-sample backtest."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence
import json
import math


class BacktestStatus(str, Enum):
    READY = "READY"
    INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class TradeResult:
    event_id: str
    ticker: str
    signal: str
    gross_return: float | None
    transaction_cost: float
    slippage: float
    net_return: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id, "ticker": self.ticker, "signal": self.signal,
            "gross_return": self.gross_return, "transaction_cost": self.transaction_cost,
            "slippage": self.slippage, "net_return": self.net_return,
        }


@dataclass(frozen=True)
class BacktestReport:
    status: BacktestStatus
    number_of_trades: int
    total_return: float | None
    annualized_return: float | None
    volatility: float | None
    sharpe_ratio: float | None
    maximum_drawdown: float | None
    win_rate: float | None
    profit_factor: float | None
    turnover: float
    trades: tuple[TradeResult, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value, "number_of_trades": self.number_of_trades,
            "total_return": self.total_return, "annualized_return": self.annualized_return,
            "volatility": self.volatility, "sharpe_ratio": self.sharpe_ratio,
            "maximum_drawdown": self.maximum_drawdown, "win_rate": self.win_rate,
            "profit_factor": self.profit_factor, "turnover": self.turnover,
            "trades": [t.to_dict() for t in self.trades], "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)


class BacktestEngine:
    """Simple event-driven backtest. No fitting, tuning, or data resplitting."""

    def run(self, trades: Sequence[TradeResult], periods_per_year: int = 252) -> BacktestReport:
        try:
            if periods_per_year <= 0:
                raise ValueError("periods_per_year must be positive")
            ordered = tuple(trades)
            active = tuple(t for t in ordered if t.signal != "NO_POSITION" and t.net_return is not None)
            if not active:
                return BacktestReport(BacktestStatus.READY, 0, 0.0, 0.0, 0.0, None, 0.0, None, None, 0.0, ordered)
            returns = [float(t.net_return) for t in active]
            equity = 1.0
            peak = 1.0
            max_dd = 0.0
            for r in returns:
                equity *= 1 + r
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak)
            total = equity - 1
            annualized = (equity ** (periods_per_year / len(returns)) - 1) if equity > 0 else None
            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / len(returns)
            volatility = math.sqrt(variance) * math.sqrt(periods_per_year)
            sharpe = (mean / math.sqrt(variance) * math.sqrt(periods_per_year)) if variance > 0 else None
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            profit_factor = (sum(wins) / abs(sum(losses))) if losses else (math.inf if wins else None)
            if profit_factor is not None and not math.isfinite(profit_factor):
                profit_factor = None
            return BacktestReport(
                BacktestStatus.READY, len(active), total, annualized, volatility, sharpe,
                max_dd, len(wins) / len(returns), profit_factor,
                float(len(active)), ordered,
            )
        except (ValueError, TypeError) as exc:
            return BacktestReport(BacktestStatus.INVALID_INPUT, 0, None, None, None, None,
                                  None, None, None, 0.0, (), str(exc))
