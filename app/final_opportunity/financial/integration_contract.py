"""Pure contract for bounded fundamental-score integration."""
from __future__ import annotations

from dataclasses import dataclass

def clamp(value, low, high):
    return max(low, min(high, float(value)))

@dataclass(frozen=True)
class CompositeScore:
    base_score: float
    fundamental_overlay: float
    final_score: float
    confidence: float
    fundamental_used: bool
    event_time_eligible: bool

    def to_dict(self):
        return self.__dict__.copy()

def combine_scores(base_score, fundamental_adjustment, *,
                   event_time_eligible=True, max_overlay=0.10):
    base = float(base_score)
    if fundamental_adjustment is None or not event_time_eligible:
        return CompositeScore(
            base, 0.0, base, 0.0, False, event_time_eligible
        )

    confidence = clamp(
        getattr(fundamental_adjustment, "confidence", 0.0), 0.0, 1.0
    )
    score = float(getattr(fundamental_adjustment, "score", 0.0))
    overlay = clamp(score, -max_overlay, max_overlay) * confidence
    return CompositeScore(
        base, overlay, base + overlay, confidence, True, True
    )
