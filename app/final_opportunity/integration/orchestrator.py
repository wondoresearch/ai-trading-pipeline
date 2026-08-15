from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class OpportunityResult:
    ticker: str
    base_score: float
    fundamental_overlay: float
    composite_score: float
    fundamental_confidence: float
    event_time_eligible: bool
    status: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d


def _clamp(x: Any, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, v))


class OpportunityOrchestrator:
    """Dependency-injected integration boundary with a fail-closed PIT guard."""

    def __init__(
        self,
        *,
        market_score: Callable[[str], float],
        fundamental_adjustment: Callable[[str], Any],
        event_time_eligible: Callable[[str], bool] | None = None,
        overlay_limit: float = 0.10,
    ) -> None:
        if overlay_limit <= 0:
            raise ValueError("overlay_limit must be positive")
        self.market_score = market_score
        self.fundamental_adjustment = fundamental_adjustment
        # Historical/PIT safety must fail closed. A caller must explicitly
        # prove event-time eligibility before fundamentals can affect ranking.
        self.event_time_eligible = event_time_eligible or (lambda ticker: False)
        self.overlay_limit = float(overlay_limit)

    def evaluate(self, ticker: str) -> OpportunityResult:
        ticker = ticker.upper().replace(".JK", "")
        base = _clamp(self.market_score(ticker), 0.0, 1.0)
        eligible = bool(self.event_time_eligible(ticker))
        raw = self.fundamental_adjustment(ticker)

        if not eligible or raw is None:
            overlay = 0.0
            confidence = 0.0
            status = "neutral_event_guard" if not eligible else "no_fundamentals"
            reasons = ("fundamental_overlay_neutral",)
        else:
            raw_score = _clamp(getattr(raw, "score", 0.0), -self.overlay_limit, self.overlay_limit)
            confidence = _clamp(getattr(raw, "confidence", 0.0), 0.0, 1.0)
            overlay = raw_score * confidence
            status = "applied" if confidence > 0 else "neutral_confidence"
            reasons = ("fundamental_overlay_applied",) if overlay else ("fundamental_overlay_neutral",)

        composite = _clamp(base + overlay, 0.0, 1.0)
        return OpportunityResult(
            ticker=ticker, base_score=base, fundamental_overlay=overlay,
            composite_score=composite, fundamental_confidence=confidence,
            event_time_eligible=eligible, status=status, reasons=reasons,
        )

    def rank(self, tickers: Iterable[str]) -> list[OpportunityResult]:
        rows = [self.evaluate(t) for t in tickers]
        return sorted(rows, key=lambda r: (-r.composite_score, r.ticker))
