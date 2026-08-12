from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Iterable, Sequence, Tuple

from scipy.stats import t as student_t

from .historical_return_data import EventStudyStatus


@dataclass(frozen=True)
class InferenceResult:
    statistic: float | None
    p_value: float | None
    rejection_at_0_05: bool | None
    sample_size: int
    average_residual_correlation: float | None
    average_overlap: float | None
    adjustment_factor: float | None
    status: EventStudyStatus

    def to_dict(self):
        return {"statistic": self.statistic, "p_value": self.p_value,
                "rejection_at_0_05": self.rejection_at_0_05, "sample_size": self.sample_size,
                "average_residual_correlation": self.average_residual_correlation,
                "average_overlap": self.average_overlap, "adjustment_factor": self.adjustment_factor,
                "status": self.status.value}


class InferenceEngine:
    MINIMUM_EVENTS = 10
    ALPHA = 0.05

    def bmp_kolari_pynnonen(self, standardized_values: Sequence[float],
                            residual_correlations: Sequence[float],
                            overlaps: Sequence[float]) -> InferenceResult:
        n = len(standardized_values)
        if n < self.MINIMUM_EVENTS:
            return InferenceResult(None, None, None, n, None, None, None, EventStudyStatus.INSUFFICIENT_CROSS_SECTION)
        if any(not isfinite(v) for v in standardized_values):
            return InferenceResult(None, None, None, n, None, None, None, EventStudyStatus.INVALID_RETURN_DATA)
        mean = sum(standardized_values) / n
        variance = sum((value - mean) ** 2 for value in standardized_values) / (n - 1)
        if variance <= 0:
            return InferenceResult(None, None, None, n, None, None, None, EventStudyStatus.MODEL_FAILURE)
        bmp = mean / sqrt(variance / n)
        rho = sum(residual_correlations) / len(residual_correlations) if residual_correlations else 0.0
        overlap = sum(overlaps) / len(overlaps) if overlaps else None
        rho = max(-1 / (n - 1) + 1e-12, min(0.999999, rho))
        factor = sqrt((1 - rho) / (1 + (n - 1) * rho))
        statistic = bmp * factor
        p_value = float(2 * student_t.sf(abs(statistic), df=n - 1))
        return InferenceResult(statistic, p_value, p_value < self.ALPHA, n, rho, overlap, factor, EventStudyStatus.OBSERVED)
