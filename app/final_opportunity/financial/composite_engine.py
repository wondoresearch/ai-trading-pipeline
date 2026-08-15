"""Pure, bounded fundamental-to-composite scoring contract.

No provider/network/filesystem calls are made here. The `fundamental` argument
is duck-typed so this module can be tested independently and remains compatible
with the existing FundamentalAdjustment contract.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

@dataclass(frozen=True)
class CompositeScore:
    base_score: float
    fundamental_adjustment: float
    confidence: float
    score: float
    fundamental_used: bool
    event_time_eligible: bool

    def to_dict(self) -> dict:
        return {
            "base_score": self.base_score,
            "fundamental_adjustment": self.fundamental_adjustment,
            "confidence": self.confidence,
            "score": self.score,
            "fundamental_used": self.fundamental_used,
            "event_time_eligible": self.event_time_eligible,
        }

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))

def combine(
    base_score: float,
    fundamental: Optional[Any],
    *,
    event_time_eligible: bool = True,
    max_overlay: float = 0.10,
) -> CompositeScore:
    base = float(base_score)
    if (
        fundamental is None
        or not event_time_eligible
        or getattr(fundamental, "feature_set", None) is None
    ):
        return CompositeScore(base, 0.0, 0.0, base, False, event_time_eligible)

    adjustment = _clamp(getattr(fundamental, "score", 0.0), -abs(max_overlay), abs(max_overlay))
    confidence = _clamp(getattr(fundamental, "confidence", 0.0), 0.0, 1.0)
    applied = adjustment * confidence

    return CompositeScore(
        base_score=base,
        fundamental_adjustment=applied,
        confidence=confidence,
        score=base + applied,
        fundamental_used=True,
        event_time_eligible=True,
    )
