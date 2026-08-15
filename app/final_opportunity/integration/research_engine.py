from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable

from .orchestrator import OpportunityOrchestrator, OpportunityResult


@dataclass(frozen=True)
class DataStatus:
    market: str = "unknown"
    news: str = "unknown"
    financial: str = "unknown"

    @property
    def healthy(self) -> bool:
        return all(v == "healthy" for v in (self.market, self.news, self.financial))

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchOpportunity:
    result: OpportunityResult
    data_status: DataStatus
    eligible: bool
    eligibility_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.result.to_dict(),
            "data_status": self.data_status.to_dict(),
            "eligible": self.eligible,
            "eligibility_reason": self.eligibility_reason,
        }


class ResearchOpportunityEngine:
    """Production-facing, dependency-injected research integration boundary.

    It deliberately does not fetch third-party data. Providers are injected so
    the same engine can run live, offline, or in a point-in-time backtest.
    """

    def __init__(
        self,
        *,
        market_score: Callable[[str], float],
        fundamental_adjustment: Callable[[str], Any],
        event_time_eligible: Callable[[str], bool],
        data_status: Callable[[str], DataStatus] | None = None,
        overlay_limit: float = 0.10,
    ) -> None:
        self.orchestrator = OpportunityOrchestrator(
            market_score=market_score,
            fundamental_adjustment=fundamental_adjustment,
            event_time_eligible=event_time_eligible,
            overlay_limit=overlay_limit,
        )
        self.data_status = data_status or (lambda _: DataStatus())

    def evaluate(self, ticker: str) -> ResearchOpportunity:
        result = self.orchestrator.evaluate(ticker)
        status = self.data_status(result.ticker)
        event_ok = result.event_time_eligible

        if not event_ok:
            eligible, reason = False, "event_time_not_eligible"
        elif status.market == "unavailable":
            eligible, reason = False, "market_unavailable"
        elif status.news == "unavailable":
            eligible, reason = False, "news_unavailable"
        elif status.financial == "unavailable" and result.status not in {"no_fundamentals", "neutral_event_guard"}:
            eligible, reason = False, "financial_unavailable"
        else:
            eligible, reason = True, "eligible"

        return ResearchOpportunity(result, status, eligible, reason)

    def rank(self, tickers: Iterable[str], top_n: int | None = None) -> list[ResearchOpportunity]:
        rows = [self.evaluate(t) for t in tickers]
        rows.sort(key=lambda r: (-r.result.composite_score, r.result.ticker))
        if top_n is not None:
            if top_n <= 0:
                raise ValueError("top_n must be positive")
            return rows[:top_n]
        return rows
