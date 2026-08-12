"""Phase 15 — OOS statistical robustness validation.

Validates a frozen OOS strategy against a zero-return null using deterministic
bootstrap confidence intervals and a sign-flip permutation test.

This module never retrains models, changes signals, selects thresholds, or
looks at in-sample data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Sequence


class RobustnessStatus(str, Enum):
    READY = "READY"
    INVALID_INPUT = "INVALID_INPUT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class RobustnessReport:
    status: RobustnessStatus
    sample_size: int
    observed_mean: float | None
    observed_total_return: float | None
    bootstrap_mean_ci_low: float | None
    bootstrap_mean_ci_high: float | None
    permutation_p_value: float | None
    significant_at_0_05: bool | None
    bootstrap_iterations: int
    permutation_iterations: int
    seed: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "sample_size": self.sample_size,
            "observed_mean": self.observed_mean,
            "observed_total_return": self.observed_total_return,
            "bootstrap_mean_ci_low": self.bootstrap_mean_ci_low,
            "bootstrap_mean_ci_high": self.bootstrap_mean_ci_high,
            "permutation_p_value": self.permutation_p_value,
            "significant_at_0_05": self.significant_at_0_05,
            "bootstrap_iterations": self.bootstrap_iterations,
            "permutation_iterations": self.permutation_iterations,
            "seed": self.seed,
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


class OOSRobustnessValidator:
    """Deterministic statistical checks over frozen OOS realized returns."""

    def validate(
        self,
        returns: Sequence[float],
        *,
        bootstrap_iterations: int = 2000,
        permutation_iterations: int = 2000,
        seed: int = 20260812,
        min_observations: int = 10,
        confidence: float = 0.95,
    ) -> RobustnessReport:
        try:
            values = tuple(float(x) for x in returns)
            if not values:
                raise ValueError("returns must not be empty")
            if not all(math.isfinite(x) for x in values):
                raise ValueError("returns must be finite")
            if bootstrap_iterations <= 0 or permutation_iterations <= 0:
                raise ValueError("iteration counts must be positive")
            if min_observations <= 0:
                raise ValueError("min_observations must be positive")
            if not 0 < confidence < 1:
                raise ValueError("confidence must be between 0 and 1")

            mean = sum(values) / len(values)
            total = math.prod(1.0 + x for x in values) - 1.0

            if len(values) < min_observations:
                return RobustnessReport(
                    RobustnessStatus.INSUFFICIENT_DATA, len(values), mean, total,
                    None, None, None, None, bootstrap_iterations,
                    permutation_iterations, seed,
                )

            boot = self._bootstrap_means(
                values, bootstrap_iterations, seed
            )
            alpha = (1.0 - confidence) / 2.0
            low = self._quantile(boot, alpha)
            high = self._quantile(boot, 1.0 - alpha)

            p_value = self._permutation_p_value(
                values, permutation_iterations, seed
            )
            return RobustnessReport(
                RobustnessStatus.READY,
                len(values),
                mean,
                total,
                low,
                high,
                p_value,
                p_value < 0.05,
                bootstrap_iterations,
                permutation_iterations,
                seed,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            return RobustnessReport(
                RobustnessStatus.INVALID_INPUT,
                0, None, None, None, None, None, None,
                bootstrap_iterations, permutation_iterations, seed,
                str(exc),
            )

    @staticmethod
    def _rng(seed: int):
        # Small deterministic PRNG, avoiding global random state.
        state = (seed ^ 0x9E3779B97F4A7C15) & ((1 << 64) - 1)

        def next_u64() -> int:
            nonlocal state
            state ^= (state << 13) & ((1 << 64) - 1)
            state ^= state >> 7
            state ^= (state << 17) & ((1 << 64) - 1)
            state &= (1 << 64) - 1
            return state

        return next_u64

    @classmethod
    def _bootstrap_means(
        cls, values: tuple[float, ...], iterations: int, seed: int
    ) -> list[float]:
        next_u64 = cls._rng(seed)
        n = len(values)
        result = []
        for _ in range(iterations):
            total = 0.0
            for _ in range(n):
                total += values[next_u64() % n]
            result.append(total / n)
        result.sort()
        return result

    @classmethod
    def _permutation_p_value(
        cls, values: tuple[float, ...], iterations: int, seed: int
    ) -> float:
        next_u64 = cls._rng(seed ^ 0xA5A5A5A5)
        observed = abs(sum(values) / len(values))
        extreme = 0
        for _ in range(iterations):
            signed_sum = 0.0
            for value in values:
                signed_sum += value if next_u64() & 1 else -value
            if abs(signed_sum / len(values)) >= observed:
                extreme += 1
        return (extreme + 1) / (iterations + 1)

    @staticmethod
    def _quantile(sorted_values: Sequence[float], q: float) -> float:
        if not sorted_values:
            raise ValueError("sorted_values must not be empty")
        position = (len(sorted_values) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return sorted_values[lower]
        fraction = position - lower
        return sorted_values[lower] + (
            sorted_values[upper] - sorted_values[lower]
        ) * fraction

    @staticmethod
    def fingerprint(returns: Sequence[float]) -> str:
        payload = json.dumps(
            [float(x) for x in returns],
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
