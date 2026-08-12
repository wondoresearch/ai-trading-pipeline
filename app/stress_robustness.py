"""Phase 18 — controlled stress and adversarial robustness analysis.

Evaluation-only layer over frozen OOS strategy returns. It does not retrain,
retune thresholds, select models, or use stressed results to alter signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Sequence


class StressScenario(str, Enum):
    BASELINE = "BASELINE"
    RETURN_SHOCK = "RETURN_SHOCK"
    SIGNAL_NOISE = "SIGNAL_NOISE"
    TRANSACTION_COST = "TRANSACTION_COST"


class StressStatus(str, Enum):
    READY = "READY"
    INVALID_INPUT = "INVALID_INPUT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class OOSObservation:
    timestamp: float
    strategy_return: float
    signal: int = 1


@dataclass(frozen=True)
class StressConfig:
    return_shock: float = 0.0
    signal_flip_rate: float = 0.0
    transaction_cost: float = 0.0


@dataclass(frozen=True)
class StressResult:
    scenario: StressScenario
    sample_size: int
    total_return: float
    mean_return: float
    maximum_drawdown: float
    degradation_vs_baseline: float
    relative_robustness: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.value,
            "sample_size": self.sample_size,
            "total_return": self.total_return,
            "mean_return": self.mean_return,
            "maximum_drawdown": self.maximum_drawdown,
            "degradation_vs_baseline": self.degradation_vs_baseline,
            "relative_robustness": self.relative_robustness,
        }


@dataclass(frozen=True)
class StressReport:
    status: StressStatus
    baseline: StressResult | None
    scenarios: tuple[StressResult, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "scenarios": [x.to_dict() for x in self.scenarios],
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )


class StressTester:
    """Apply deterministic, caller-defined perturbations to frozen OOS data."""

    def run(
        self,
        observations: Sequence[OOSObservation],
        configs: Sequence[StressConfig],
    ) -> StressReport:
        try:
            obs = tuple(observations)
            self._validate(obs)
            if not configs:
                return StressReport(
                    StressStatus.INSUFFICIENT_DATA, None, (),
                    "at least one stress configuration is required",
                )
            self._validate_configs(configs)

            baseline = self._result(
                StressScenario.BASELINE, obs,
                StressConfig(),
                baseline_total=None,
            )
            results: list[StressResult] = []
            seen: set[tuple[float, float, float]] = set()

            for config in configs:
                key = (
                    float(config.return_shock),
                    float(config.signal_flip_rate),
                    float(config.transaction_cost),
                )
                if key in seen:
                    continue
                seen.add(key)
                scenario = self._scenario_name(config)
                result = self._result(
                    scenario, obs, config,
                    baseline_total=baseline.total_return,
                )
                results.append(result)

            return StressReport(
                StressStatus.READY, baseline, tuple(results)
            )
        except (ValueError, TypeError, IndexError, OverflowError) as exc:
            return StressReport(StressStatus.INVALID_INPUT, None, (), str(exc))

    @staticmethod
    def _scenario_name(config: StressConfig) -> StressScenario:
        active = sum([
            config.return_shock != 0,
            config.signal_flip_rate != 0,
            config.transaction_cost != 0,
        ])
        if active != 1:
            # Mixed stress is deliberately represented as SIGNAL_NOISE because
            # the result remains an evaluation artifact, not a model-selection
            # label. The configuration itself is preserved by the values.
            return StressScenario.SIGNAL_NOISE
        if config.return_shock != 0:
            return StressScenario.RETURN_SHOCK
        if config.signal_flip_rate != 0:
            return StressScenario.SIGNAL_NOISE
        return StressScenario.TRANSACTION_COST

    @classmethod
    def _result(
        cls,
        scenario: StressScenario,
        observations: Sequence[OOSObservation],
        config: StressConfig,
        baseline_total: float | None,
    ) -> StressResult:
        returns: list[float] = []
        flip = config.signal_flip_rate > 0
        for i, obs in enumerate(observations):
            r = float(obs.strategy_return)
            if config.return_shock:
                r *= 1.0 - config.return_shock
            if flip:
                # Deterministic pseudo-random-free perturbation: every
                # reciprocal interval defined by the configured rate flips.
                period = max(1, round(1.0 / config.signal_flip_rate))
                if (i + 1) % period == 0:
                    r = -r
            if config.transaction_cost:
                r -= abs(obs.signal) * config.transaction_cost
            returns.append(r)

        total = math.prod(1.0 + r for r in returns) - 1.0
        mean = sum(returns) / len(returns)
        equity = peak = 1.0
        max_dd = 0.0
        for r in returns:
            equity *= 1.0 + r
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / peak - 1.0)
        degradation = (
            0.0 if baseline_total is None else total - baseline_total
        )
        robustness = (
            None if baseline_total is None or baseline_total == 0
            else total / baseline_total
        )
        return StressResult(
            scenario, len(returns), total, mean, max_dd,
            degradation, robustness,
        )

    @staticmethod
    def _validate(observations: Sequence[OOSObservation]) -> None:
        if not observations:
            raise ValueError("observations must not be empty")
        timestamps = [x.timestamp for x in observations]
        if timestamps != sorted(timestamps):
            raise ValueError("observations must be chronologically sorted")
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("timestamps must be unique")
        for x in observations:
            if not math.isfinite(float(x.strategy_return)):
                raise ValueError("strategy_return must be finite")
            if x.signal not in (-1, 0, 1):
                raise ValueError("signal must be -1, 0, or 1")

    @staticmethod
    def _validate_configs(configs: Sequence[StressConfig]) -> None:
        for x in configs:
            if not 0 <= x.return_shock < 1:
                raise ValueError("return_shock must be in [0, 1)")
            if not 0 <= x.signal_flip_rate <= 1:
                raise ValueError("signal_flip_rate must be in [0, 1]")
            if x.transaction_cost < 0:
                raise ValueError("transaction_cost must be non-negative")
