from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from .event_schema import NewsEvent
from .market_calendar import IDXMarketCalendar, MarketSession
from .time_utils import parse_datetime


class EventTimeRule(str, Enum):
    REGULAR_SESSION = "regular_session"
    PRE_MARKET_TO_OPEN = "pre_market_to_open"
    LUNCH_BREAK_TO_SESSION_II = "lunch_break_to_session_ii"
    PRE_CLOSING = "pre_closing"
    POST_CLOSING_TO_NEXT_OPEN = "post_closing_to_next_open"
    AFTER_MARKET_TO_NEXT_OPEN = "after_market_to_next_open"
    NON_TRADING_DAY_TO_NEXT_OPEN = "non_trading_day_to_next_open"
    CLOSED_BEFORE_PRE_MARKET_TO_NEXT_OPEN = "closed_before_pre_market_to_next_open"


@dataclass(frozen=True)
class EventTimeInput:
    """Validated boundary object for the event-time engine."""

    event_id: str
    ticker: str
    event_time: datetime

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not self.ticker:
            raise ValueError("ticker must not be empty")
        if self.event_time.tzinfo is None or self.event_time.utcoffset() is None:
            raise ValueError("event_time must be timezone-aware")


@dataclass(frozen=True)
class EventTimeResolution:
    """Resolved market-time semantics for one ticker-level event."""

    event_id: str
    ticker: str
    event_time: datetime
    market_date: date
    market_session: MarketSession
    effective_time: datetime
    resolution_rule: EventTimeRule
    is_trading_day: bool
    is_tradeable_at_event: bool
    is_same_session_effective: bool

    def __post_init__(self) -> None:
        if self.event_time.tzinfo is None or self.event_time.utcoffset() is None:
            raise ValueError("event_time must be timezone-aware")
        if self.effective_time.tzinfo is None or self.effective_time.utcoffset() is None:
            raise ValueError("effective_time must be timezone-aware")
        if self.effective_time < self.event_time:
            raise ValueError("effective_time cannot be earlier than event_time")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "event_time": self.event_time.isoformat(),
            "market_date": self.market_date.isoformat(),
            "market_session": self.market_session.value,
            "effective_time": self.effective_time.isoformat(),
            "resolution_rule": self.resolution_rule.value,
            "is_trading_day": self.is_trading_day,
            "is_tradeable_at_event": self.is_tradeable_at_event,
            "is_same_session_effective": self.is_same_session_effective,
        }


class IDXEventTimeEngine:
    """Resolve event timestamps against the frozen IDX market calendar.

    Phase 3 stops at effective_time. It does not fetch prices, calculate
    returns, or select a price observation. Those responsibilities belong
    to Phase 4.
    """

    TIMEZONE = "Asia/Jakarta"

    def __init__(self, calendar: Optional[IDXMarketCalendar] = None) -> None:
        self.calendar = calendar or IDXMarketCalendar()
        self.tz = ZoneInfo(self.TIMEZONE)

    def resolve(self, event: EventTimeInput) -> EventTimeResolution:
        local_time = self._to_local(event.event_time)
        market_date = local_time.date()
        is_trading_day = self.calendar.is_trading_day(market_date)

        if not is_trading_day:
            return self._build(
                event=event,
                local_time=local_time,
                market_date=market_date,
                session=MarketSession.CLOSED,
                effective_time=self._next_open_after(market_date),
                rule=EventTimeRule.NON_TRADING_DAY_TO_NEXT_OPEN,
                is_trading_day=False,
                is_tradeable=False,
            )

        session = self.calendar.session_at(local_time)

        if session == MarketSession.CLOSED:
            return self._build(
                event=event,
                local_time=local_time,
                market_date=market_date,
                session=session,
                effective_time=self._next_open_after(market_date),
                rule=EventTimeRule.CLOSED_BEFORE_PRE_MARKET_TO_NEXT_OPEN,
                is_trading_day=True,
                is_tradeable=False,
            )

        if session == MarketSession.PRE_MARKET:
            return self._build(
                event=event,
                local_time=local_time,
                market_date=market_date,
                session=session,
                effective_time=self.calendar.regular_open(market_date),
                rule=EventTimeRule.PRE_MARKET_TO_OPEN,
                is_trading_day=True,
                is_tradeable=False,
            )

        if session == MarketSession.REGULAR:
            return self._build(
                event=event,
                local_time=local_time,
                market_date=market_date,
                session=session,
                effective_time=local_time,
                rule=EventTimeRule.REGULAR_SESSION,
                is_trading_day=True,
                is_tradeable=True,
            )

        if session == MarketSession.LUNCH_BREAK:
            return self._build(
                event=event,
                local_time=local_time,
                market_date=market_date,
                session=session,
                effective_time=self.calendar.session_2_start(market_date),
                rule=EventTimeRule.LUNCH_BREAK_TO_SESSION_II,
                is_trading_day=True,
                is_tradeable=False,
            )

        if session == MarketSession.PRE_CLOSING:
            return self._build(
                event=event,
                local_time=local_time,
                market_date=market_date,
                session=session,
                effective_time=local_time,
                rule=EventTimeRule.PRE_CLOSING,
                is_trading_day=True,
                is_tradeable=True,
            )

        if session == MarketSession.POST_CLOSING:
            return self._build(
                event=event,
                local_time=local_time,
                market_date=market_date,
                session=session,
                effective_time=self._next_open_after(market_date),
                rule=EventTimeRule.POST_CLOSING_TO_NEXT_OPEN,
                is_trading_day=True,
                is_tradeable=False,
            )

        if session == MarketSession.AFTER_MARKET:
            return self._build(
                event=event,
                local_time=local_time,
                market_date=market_date,
                session=session,
                effective_time=self._next_open_after(market_date),
                rule=EventTimeRule.AFTER_MARKET_TO_NEXT_OPEN,
                is_trading_day=True,
                is_tradeable=False,
            )

        raise ValueError(f"Unsupported market session: {session!r}")

    def resolve_news_event(self, event: NewsEvent) -> EventTimeResolution:
        """Resolve a NewsEvent using its canonical UTC publication timestamp."""
        parsed = parse_datetime(event.published_at_utc)
        if parsed is None:
            raise ValueError(
                f"NewsEvent {event.event_id} has no valid published_at_utc"
            )
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(
                f"NewsEvent {event.event_id} published_at_utc must be timezone-aware"
            )

        return self.resolve(
            EventTimeInput(
                event_id=event.event_id,
                ticker=event.ticker,
                event_time=parsed,
            )
        )

    def _next_open_after(self, market_date: date) -> datetime:
        next_day = self.calendar.next_trading_day(market_date)
        return self.calendar.regular_open(next_day)

    def _to_local(self, value: datetime) -> datetime:
        return value.astimezone(self.tz)

    @staticmethod
    def _build(
        *,
        event: EventTimeInput,
        local_time: datetime,
        market_date: date,
        session: MarketSession,
        effective_time: datetime,
        rule: EventTimeRule,
        is_trading_day: bool,
        is_tradeable: bool,
    ) -> EventTimeResolution:
        effective_time = effective_time.astimezone(local_time.tzinfo)
        return EventTimeResolution(
            event_id=event.event_id,
            ticker=event.ticker,
            event_time=local_time,
            market_date=market_date,
            market_session=session,
            effective_time=effective_time,
            resolution_rule=rule,
            is_trading_day=is_trading_day,
            is_tradeable_at_event=is_tradeable,
            is_same_session_effective=effective_time.date() == market_date,
        )
