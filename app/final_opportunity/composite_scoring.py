"""Production-safe composite scoring adapter.

Keeps the existing market/news score as the base signal and applies the
already-reviewed fundamental integration contract as a bounded overlay.
No external provider calls occur in this layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .financial.integration_contract import combine_scores


@dataclass(frozen=True)
class CompositeOpportunity:
    ticker: str
    base_score: float
    fundamental_overlay: float
    final_score: float
    fundamental_confidence: float
    fundamental_used: bool
    event_time_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "base_score": self.base_score,
            "fundamental_overlay": self.fundamental_overlay,
            "final_score": self.final_score,
            "fundamental_confidence": self.fundamental_confidence,
            "fundamental_used": self.fundamental_used,
            "event_time_eligible": self.event_time_eligible,
        }


def integrate_fundamentals(
    ticker: str,
    base_score: float,
    fundamental_adjustment: Optional[Any],
    *,
    event_time_eligible: bool = True,
) -> CompositeOpportunity:
    """Apply the bounded fundamental overlay without changing base scoring."""
    result = combine_scores(
        base_score,
        fundamental_adjustment,
        event_time_eligible=event_time_eligible,
    )
    return CompositeOpportunity(
        ticker=ticker,
        base_score=result.base_score,
        fundamental_overlay=result.fundamental_overlay,
        final_score=result.final_score,
        fundamental_confidence=result.confidence,
        fundamental_used=result.fundamental_used,
        event_time_eligible=result.event_time_eligible,
    )
