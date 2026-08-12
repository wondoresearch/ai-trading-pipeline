"""Phase 17 — leakage-safe market-regime robustness analysis.

Regimes are classified from information available at or before each observation.
The analyzer evaluates already-generated OOS strategy returns by regime and
does not retrain models, tune thresholds, or inspect future returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Sequence


class Regime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"


class RegimeAnalysisStatus(str, Enum):
    READY = "READY"
    INVALID_INPUT = "INVALID_INPUT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class RegimeObservation:
    timestamp: float
    benchmark_return: float
    strategy_return: float


@dataclass(frozen=True)
class RegimeMetrics:
    regime: Regime
    sample_size: int
    total_return: float
    mean_return: float
    volatility: float
    hit_rate: float
    maximum_drawdown: float
    sharpe_like: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "sample_size": self.sample_size,
            "total_return": self.total_return,
            "mean_return": self.mean_return,
            "volatility": self.volatility,
            "hit_rate": self.hit_rate,
            "maximum_drawdown": self.maximum_drawdown,
            "sharpe_like": self.sharpe_like,
        }


@dataclass(frozen=True)
class RegimeAnalysisReport:
    status: RegimeAnalysisStatus
    metrics: tuple[RegimeMetrics, ...]
    regime_counts: dict[str, int]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "metrics": [m.to_dict() for m in self.metrics],
            "regime_counts": dict(sorted(self.regime_counts.items())),
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


class MarketRegimeClassifier:
    """Classify using trailing benchmark returns only.

    For index i, the signal is the mean benchmark return over observations
    strictly before i. The first ``lookback`` observations are therefore
    SIDEWAYS because insufficient history exists.
    """

    def __init__(
        self,
        lookback: int = 5,
        bull_threshold: float = 0.002,
        bear_threshold: float = -0.002,
    ) -> None:
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        if bear_threshold >= bull_threshold:
            raise ValueError("bear_threshold must be below bull_threshold")
        self.lookback = lookback
        self.bull_threshold = float(bull_threshold)
        self.bear_threshold = float(bear_threshold)

    def classify(self, observations: Sequence[RegimeObservation]) -> tuple[Regime, ...]:
        if not observations:
            raise ValueError("observations must not be empty")
        result: list[Regime] = []
        history: list[float] = []
        for obs in observations:
            if len(history) < self.lookback:
                result.append(Regime.SIDEWAYS)
            else:
                trailing_mean = sum(history[-self.lookback:]) / self.lookback
                if trailing_mean >= self.bull_threshold:
                    result.append(Regime.BULL)
                elif trailing_mean <= self.bear_threshold:
                    result.append(Regime.BEAR)
                else:
                    result.append(Regime.SIDEWAYS)
            history.append(float(obs.benchmark_return))
        return tuple(result)


class RegimeRobustnessAnalyzer:
    """Measure strategy behavior conditional on precomputed regime labels."""

    def analyze(
        self,
        observations: Sequence[RegimeObservation],
        regimes: Sequence[Regime] | None = None,
    ) -> RegimeAnalysisReport:
        try:
            obs = tuple(observations)
            self._validate(obs)
            labels = (
                MarketRegimeClassifier().classify(obs)
                if regimes is None else tuple(regimes)
            )
            if len(labels) != len(obs):
                raise ValueError("regime labels must match observations")
            if any(not isinstance(x, Regime) for x in labels):
                raise ValueError("regime labels must be Regime values")

            grouped: dict[Regime, list[float]] = {r: [] for r in Regime}
            for label, item in zip(labels, obs):
                grouped[label].append(float(item.strategy_return))

            metrics: list[RegimeMetrics] = []
            for regime in Regime:
                values = grouped[regime]
                if not values:
                    continue
                total = math.prod(1.0 + x for x in values) - 1.0
                mean = sum(values) / len(values)
                variance = (
                    sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                    if len(values) > 1 else 0.0
                )
                volatility = math.sqrt(variance)
                hit_rate = sum(x > 0 for x in values) / len(values)
                equity = 1.0
                peak = 1.0
                max_dd = 0.0
                for value in values:
                    equity *= 1.0 + value
                    peak = max(peak, equity)
                    max_dd = min(max_dd, equity / peak - 1.0)
                sharpe = (
                    mean / volatility * math.sqrt(len(values))
                    if volatility > 0 else None
                )
                metrics.append(
                    RegimeMetrics(
                        regime, len(values), total, mean, volatility,
                        hit_rate, max_dd, sharpe,
                    )
                )

            if not metrics:
                return RegimeAnalysisReport(
                    RegimeAnalysisStatus.INSUFFICIENT_DATA, (), {},
                    "no regime observations available",
                )

            return RegimeAnalysisReport(
                RegimeAnalysisStatus.READY,
                tuple(metrics),
                {r.value: len(grouped[r]) for r in Regime},
            )
        except (ValueError, TypeError, IndexError, OverflowError) as exc:
            return RegimeAnalysisReport(
                RegimeAnalysisStatus.INVALID_INPUT, (), {}, str(exc)
            )

    @staticmethod
    def _validate(observations: Sequence[RegimeObservation]) -> None:
        if not observations:
            raise ValueError("observations must not be empty")
        timestamps = [x.timestamp for x in observations]
        if timestamps != sorted(timestamps):
            raise ValueError("observations must be chronologically sorted")
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("timestamps must be unique")
        for item in observations:
            if not math.isfinite(float(item.benchmark_return)):
                raise ValueError("benchmark_return must be finite")
            if not math.isfinite(float(item.strategy_return)):
                raise ValueError("strategy_return must be finite")
