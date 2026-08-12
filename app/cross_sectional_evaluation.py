"""Phase 19 - cross-sectional prediction evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class CrossSectionalMetrics:
    rank_ic: float
    top_k_mean_return: float
    top_k_mean_excess_return: float
    hit_rate: float
    universe_mean_return: float
    universe_mean_excess_return: float


class CrossSectionalEvaluator:
    """Evaluate whether predicted returns rank future opportunities."""

    def evaluate(
        self,
        predicted_returns: Sequence[float],
        realized_returns: Sequence[float],
        benchmark_returns: Sequence[float],
        top_k: int = 3,
    ) -> CrossSectionalMetrics:
        self._validate(
            predicted_returns,
            realized_returns,
            benchmark_returns,
            top_k,
        )

        excess_returns = [
            realized - benchmark
            for realized, benchmark in zip(
                realized_returns,
                benchmark_returns,
            )
        ]

        ranked_indices = sorted(
            range(len(predicted_returns)),
            key=lambda index: predicted_returns[index],
            reverse=True,
        )

        selected = ranked_indices[:top_k]

        top_k_returns = [
            realized_returns[index]
            for index in selected
        ]

        top_k_excess = [
            excess_returns[index]
            for index in selected
        ]

        hit_rate = sum(
            realized_returns[index] > 0
            for index in selected
        ) / len(selected)

        return CrossSectionalMetrics(
            rank_ic=self._spearman(
                predicted_returns,
                realized_returns,
            ),
            top_k_mean_return=self._mean(top_k_returns),
            top_k_mean_excess_return=self._mean(top_k_excess),
            hit_rate=float(hit_rate),
            universe_mean_return=self._mean(realized_returns),
            universe_mean_excess_return=self._mean(excess_returns),
        )

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _spearman(
        predicted: Sequence[float],
        realized: Sequence[float],
    ) -> float:
        predicted_ranks = CrossSectionalEvaluator._ranks(
            predicted
        )
        realized_ranks = CrossSectionalEvaluator._ranks(
            realized
        )

        return CrossSectionalEvaluator._pearson(
            predicted_ranks,
            realized_ranks,
        )

    @staticmethod
    def _pearson(
        first: Sequence[float],
        second: Sequence[float],
    ) -> float:
        first_mean = CrossSectionalEvaluator._mean(first)
        second_mean = CrossSectionalEvaluator._mean(second)

        numerator = sum(
            (a - first_mean) * (b - second_mean)
            for a, b in zip(first, second)
        )

        first_var = sum(
            (a - first_mean) ** 2
            for a in first
        )

        second_var = sum(
            (b - second_mean) ** 2
            for b in second
        )

        denominator = math.sqrt(
            first_var * second_var
        )

        if denominator == 0:
            return 0.0

        return numerator / denominator

    @staticmethod
    def _ranks(values: Sequence[float]) -> list[float]:
        ordered = sorted(
            enumerate(values),
            key=lambda item: item[1],
        )

        ranks = [0.0] * len(values)
        position = 0

        while position < len(ordered):
            end = position + 1

            while (
                end < len(ordered)
                and ordered[end][1] == ordered[position][1]
            ):
                end += 1

            average_rank = (
                position + 1 + end
            ) / 2.0

            for index in range(position, end):
                ranks[ordered[index][0]] = average_rank

            position = end

        return ranks

    @staticmethod
    def _validate(
        predicted_returns,
        realized_returns,
        benchmark_returns,
        top_k,
    ) -> None:
        if not predicted_returns:
            raise ValueError("prediction data must not be empty")

        if len(predicted_returns) != len(realized_returns):
            raise ValueError(
                "predicted/realized length mismatch"
            )

        if len(predicted_returns) != len(benchmark_returns):
            raise ValueError(
                "predicted/benchmark length mismatch"
            )

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if top_k > len(predicted_returns):
            raise ValueError(
                "top_k cannot exceed universe size"
            )

        for values in (
            predicted_returns,
            realized_returns,
            benchmark_returns,
        ):
            if not all(
                math.isfinite(float(value))
                for value in values
            ):
                raise ValueError(
                    "returns must contain finite values"
                )
