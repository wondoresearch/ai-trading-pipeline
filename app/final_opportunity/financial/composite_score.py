"""Combine market/news opportunity score with sector-aware fundamentals.

The fundamental layer is deliberately an overlay:
- it cannot create an opportunity from zero market/news evidence;
- it is bounded to avoid dominating the baseline score;
- missing fundamentals reduce confidence rather than becoming zero-valued metrics;
- event-time eligibility is required before fundamentals affect the score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .enrichment import FundamentalAdjustment


@dataclass(frozen=True)
class CompositeScore:
    base_score: float
    fundamental_adjustment: float
    score: float
    confidence: float
    fundamental_used: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "base_score": self.base_score,
            "fundamental_adjustment": self.fundamental_adjustment,
            "score": self.score,
            "confidence": self.confidence,
            "fundamental_used": self.fundamental_used,
            "reason": self.reason,
        }


def combine_scores(
    base_score: float,
    fundamental: Optional[FundamentalAdjustment],
    *,
    event_time_eligible: bool = True,
    max_adjustment: float = 0.10,
) -> CompositeScore:
    """Return a bounded composite score.

    ``max_adjustment`` is an absolute score overlay. A missing or
    event-time-ineligible fundamental observation is neutral and never
    treated as a negative financial signal.
    """
    base = float(base_score)

    if fundamental is None or fundamental.feature_set is None:
        return CompositeScore(
            base_score=base,
            fundamental_adjustment=0.0,
            score=base,
            confidence=0.0,
            fundamental_used=False,
            reason="fundamental_unavailable",
        )

    if not event_time_eligible:
        return CompositeScore(
            base_score=base,
            fundamental_adjustment=0.0,
            score=base,
            confidence=0.0,
            fundamental_used=False,
            reason="fundamental_not_event_time_eligible",
        )

    bounded = max(-max_adjustment, min(max_adjustment, float(fundamental.score)))
    score = base + bounded
    confidence = max(0.0, min(1.0, float(fundamental.confidence)))

    return CompositeScore(
        base_score=base,
        fundamental_adjustment=bounded,
        score=score,
        confidence=confidence,
        fundamental_used=True,
        reason="sector_aware_fundamental_overlay",
    )
