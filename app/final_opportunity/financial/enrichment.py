"""Optional bridge from financial fundamentals into opportunity scoring.

This module does not replace the existing market/news score. It produces an
additive fundamental component so the existing research pipeline remains
backward compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .features import build_features
from .models import FinancialFeatureSet, FinancialObservation


@dataclass(frozen=True)
class FundamentalAdjustment:
    score: float
    confidence: float
    feature_set: Optional[FinancialFeatureSet]

    def to_dict(self):
        return {
            "score": self.score,
            "confidence": self.confidence,
            "features": self.feature_set.to_dict() if self.feature_set else None,
        }


def fundamental_adjustment(
    current: Optional[FinancialObservation],
    previous: Optional[FinancialObservation] = None,
) -> FundamentalAdjustment:
    if current is None:
        return FundamentalAdjustment(0.0, 0.0, None)

    features = build_features([current], previous)
    # Keep fundamentals as a bounded overlay, not the dominant trading signal.
    adjustment = (features.financial_score - 0.5) * 0.20
    return FundamentalAdjustment(
        score=adjustment,
        confidence=features.quality,
        feature_set=features,
    )
