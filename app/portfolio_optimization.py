"""Phase 21 — constrained portfolio optimization.

Optimizes portfolio weights using explicitly supplied *training/estimation*
statistics only. OOS returns are intentionally absent from this API.

The optimizer uses deterministic projected-gradient ascent for a
mean-variance objective under long-only, per-ticker, gross, and net exposure
constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Sequence


class OptimizationStatus(str, Enum):
    READY = "READY"
    INVALID_INPUT = "INVALID_INPUT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class OptimizationConfig:
    risk_aversion: float = 1.0
    max_weight: float = 0.50
    max_iterations: int = 500
    learning_rate: float = 0.05
    tolerance: float = 1e-10


@dataclass(frozen=True)
class OptimizationResult:
    status: OptimizationStatus
    tickers: tuple[str, ...]
    weights: tuple[float, ...]
    objective: float | None
    expected_return: float | None
    portfolio_variance: float | None
    iterations: int
    converged: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "tickers": list(self.tickers),
            "weights": list(self.weights),
            "objective": self.objective,
            "expected_return": self.expected_return,
            "portfolio_variance": self.portfolio_variance,
            "iterations": self.iterations,
            "converged": self.converged,
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )


class PortfolioOptimizer:
    """Optimize weights from supplied estimation-period statistics only."""

    def optimize(
        self,
        tickers: Sequence[str],
        expected_returns: Sequence[float],
        covariance: Sequence[Sequence[float]],
        config: OptimizationConfig,
    ) -> OptimizationResult:
        try:
            names = tuple(tickers)
            mu = tuple(float(x) for x in expected_returns)
            cov = tuple(tuple(float(x) for x in row) for row in covariance)
            self._validate(names, mu, cov, config)

            n = len(names)
            weights = [1.0 / n] * n
            weights = self._project(weights, config.max_weight)

            converged = False
            iterations = 0

            for iteration in range(1, config.max_iterations + 1):
                gradient = self._gradient(weights, mu, cov, config.risk_aversion)
                candidate = [
                    w + config.learning_rate * g
                    for w, g in zip(weights, gradient)
                ]
                candidate = self._project(candidate, config.max_weight)
                delta = max(abs(a - b) for a, b in zip(candidate, weights))
                weights = candidate
                iterations = iteration
                if delta <= config.tolerance:
                    converged = True
                    break

            expected = sum(w * m for w, m in zip(weights, mu))
            variance = self._quadratic(weights, cov)
            objective = expected - config.risk_aversion * variance
            return OptimizationResult(
                OptimizationStatus.READY,
                names,
                tuple(weights),
                objective,
                expected,
                variance,
                iterations,
                converged,
            )
        except (ValueError, TypeError, OverflowError, IndexError) as exc:
            return OptimizationResult(
                OptimizationStatus.INVALID_INPUT,
                tuple(tickers),
                (),
                None, None, None, 0, False, str(exc),
            )

    @staticmethod
    def _gradient(
        weights: Sequence[float],
        mu: Sequence[float],
        cov: Sequence[Sequence[float]],
        risk_aversion: float,
    ) -> list[float]:
        return [
            mu[i] - 2.0 * risk_aversion *
            sum(cov[i][j] * weights[j] for j in range(len(weights)))
            for i in range(len(weights))
        ]

    @staticmethod
    def _quadratic(
        weights: Sequence[float],
        cov: Sequence[Sequence[float]],
    ) -> float:
        return sum(
            weights[i] * cov[i][j] * weights[j]
            for i in range(len(weights))
            for j in range(len(weights))
        )

    @staticmethod
    def _project(
        weights: Sequence[float],
        max_weight: float,
    ) -> list[float]:
        """Project onto {w: sum(w)=1, 0 <= w_i <= max_weight}."""
        values = [float(w) for w in weights]

        if not values:
            raise ValueError("weights must not be empty")

        if max_weight * len(values) < 1.0 - 1e-12:
            raise ValueError("max_weight is too small to fully invest")

        # Find lambda such that:
        # sum(clip(x_i - lambda, 0, max_weight)) == 1.
        lo = min(values) - max_weight
        hi = max(values)

        for _ in range(100):
            mid = (lo + hi) / 2.0
            projected = [
                min(max(x - mid, 0.0), max_weight)
                for x in values
            ]

            if sum(projected) > 1.0:
                lo = mid
            else:
                hi = mid

        result = [
            min(max(x - hi, 0.0), max_weight)
            for x in values
        ]

        # Deterministic numerical correction while respecting the cap.
        residual = 1.0 - sum(result)

        if abs(residual) > 1e-10:
            if residual > 0:
                for i in range(len(result)):
                    room = max_weight - result[i]
                    delta = min(room, residual)
                    result[i] += delta
                    residual -= delta
                    if residual <= 1e-12:
                        break
            else:
                for i in range(len(result)):
                    delta = min(result[i], -residual)
                    result[i] -= delta
                    residual += delta
                    if residual >= -1e-12:
                        break

        if (
            abs(sum(result) - 1.0) > 1e-8
            or any(w < -1e-10 or w > max_weight + 1e-10 for w in result)
        ):
            raise ValueError("unable to satisfy weight constraints")

        return result

    @staticmethod
    def _validate(
        tickers: Sequence[str],
        mu: Sequence[float],
        cov: Sequence[Sequence[float]],
        config: OptimizationConfig,
    ) -> None:
        n = len(tickers)
        if n == 0:
            raise ValueError("tickers must not be empty")
        if len(mu) != n or len(cov) != n:
            raise ValueError("dimensions must agree")
        if len(set(tickers)) != n or any(not x for x in tickers):
            raise ValueError("tickers must be unique and non-empty")
        if not 0 < config.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if config.max_weight * n < 1.0 - 1e-12:
            raise ValueError("max_weight is too small to fully invest")
        if config.risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")
        if config.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if config.learning_rate <= 0 or not math.isfinite(config.learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if config.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        for x in mu:
            if not math.isfinite(x):
                raise ValueError("expected returns must be finite")
        for i, row in enumerate(cov):
            if len(row) != n:
                raise ValueError("covariance must be square")
            for x in row:
                if not math.isfinite(x):
                    raise ValueError("covariance must be finite")
            if cov[i][i] < 0:
                raise ValueError("variance must be non-negative")
        for i in range(n):
            for j in range(n):
                if abs(cov[i][j] - cov[j][i]) > 1e-8:
                    raise ValueError("covariance must be symmetric")
