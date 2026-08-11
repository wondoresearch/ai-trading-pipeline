from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

import exchange_calendars as xcals


class MarketSession(str, Enum):
    """IDX regular-market time states."""

    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    LUNCH_BREAK = "lunch_break"
    PRE_CLOSING = "pre_closing"
    POST_CLOSING = "post_closing"
    AFTER_MARKET = "after_market"


@dataclass(frozen=True)
class MarketContext:
    """Complete IDX market context for one local calendar date.

    Datetime fields are timezone-aware ``datetime`` objects inside the
    application domain. ``to_dict()`` is the serialization boundary and
    converts them to ISO-8601 strings.
    """

    market: str
    timezone: str
    calendar: str
    local_date: str
    weekday: str
    is_trading_day: bool
    session: MarketSession

    pre_market_start: Optional[datetime]
    regular_open: Optional[datetime]
    session_1_end: Optional[datetime]
    session_2_start: Optional[datetime]
    regular_close: Optional[datetime]
    pre_closing_end: Optional[datetime]
    post_closing_end: Optional[datetime]

    next_trading_day: Optional[str]
    previous_trading_day: Optional[str]

    @staticmethod
    def _iso(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value is not None else None

    def to_dict(self) -> dict:
        return {
            "market": self.market,
            "timezone": self.timezone,
            "calendar": self.calendar,
            "local_date": self.local_date,
            "weekday": self.weekday,
            "is_trading_day": self.is_trading_day,
            "session": self.session.value,
            "pre_market_start": self._iso(self.pre_market_start),
            "regular_open": self._iso(self.regular_open),
            "session_1_end": self._iso(self.session_1_end),
            "session_2_start": self._iso(self.session_2_start),
            "regular_close": self._iso(self.regular_close),
            "pre_closing_end": self._iso(self.pre_closing_end),
            "post_closing_end": self._iso(self.post_closing_end),
            "next_trading_day": self.next_trading_day,
            "previous_trading_day": self.previous_trading_day,
        }


class IDXMarketCalendar:
    """IDX equity regular-market calendar.

    Trading-day/holiday membership is delegated to the XIDX calendar from
    ``exchange_calendars``.

    Intraday session rules are defined explicitly here from IDX regular-market
    trading rules. This intentionally separates the trading-day calendar from
    the intraday session semantics.

    Mon-Thu:
        pre-market start 08:45
        regular session I 09:00-12:00
        lunch break 12:00-13:30
        regular session II 13:30-15:50
        pre-closing 15:50-16:02
        post-closing 16:02-16:15

    Friday:
        pre-market start 08:45
        regular session I 09:00-11:30
        lunch break 11:30-14:00
        regular session II 14:00-15:50
        pre-closing 15:50-16:02
        post-closing 16:02-16:15

    Boundary convention:
        start <= t < end
        except POST_CLOSING, which includes 16:15:00.
    """

    MARKET = "IDX"
    CALENDAR = "XIDX"
    TIMEZONE = "Asia/Jakarta"

    PRE_MARKET_START = time(8, 45)
    REGULAR_OPEN = time(9, 0)

    MON_THU_SESSION_1_END = time(12, 0)
    MON_THU_SESSION_2_START = time(13, 30)

    FRI_SESSION_1_END = time(11, 30)
    FRI_SESSION_2_START = time(14, 0)

    REGULAR_CLOSE = time(15, 50)
    PRE_CLOSING_END = time(16, 2)
    POST_CLOSING_END = time(16, 15)

    def __init__(self):
        self.tz = ZoneInfo(self.TIMEZONE)
        self.calendar = xcals.get_calendar(self.CALENDAR)

    def _as_date(self, value) -> date:
        if isinstance(value, datetime):
            return value.astimezone(self.tz).date() if value.tzinfo else value.date()
        if isinstance(value, date):
            return value
        raise TypeError("Expected date or datetime")

    def _as_local_datetime(self, value: datetime) -> datetime:
        if not isinstance(value, datetime):
            raise TypeError("Expected datetime")
        if value.tzinfo is None:
            # Naive datetimes are interpreted as local IDX time at this API
            # boundary. Upstream news timestamps should be normalized before
            # reaching this class.
            return value.replace(tzinfo=self.tz)
        return value.astimezone(self.tz)

    @staticmethod
    def _combine(d: date, t: time, tz: ZoneInfo) -> datetime:
        return datetime.combine(d, t, tzinfo=tz)

    def is_trading_day(self, value) -> bool:
        d = self._as_date(value)
        return bool(self.calendar.is_session(d.isoformat()))

    def _find_next_trading_day(self, value: date) -> date:
        candidate = value + timedelta(days=1)
        for _ in range(370):
            if self.is_trading_day(candidate):
                return candidate
            candidate += timedelta(days=1)
        raise RuntimeError("Could not find next IDX trading day")

    def _find_previous_trading_day(self, value: date) -> date:
        candidate = value - timedelta(days=1)
        for _ in range(370):
            if self.is_trading_day(candidate):
                return candidate
            candidate -= timedelta(days=1)
        raise RuntimeError("Could not find previous IDX trading day")

    def next_trading_day(self, value) -> date:
        d = self._as_date(value)
        if not self.is_trading_day(d):
            return self._find_next_trading_day(d)
        return self.calendar.next_session(d.isoformat()).date()

    def previous_trading_day(self, value) -> date:
        d = self._as_date(value)
        if not self.is_trading_day(d):
            return self._find_previous_trading_day(d)
        return self.calendar.previous_session(d.isoformat()).date()

    def trading_days_between(self, start, end):
        start_date = self._as_date(start)
        end_date = self._as_date(end)
        sessions = self.calendar.sessions_in_range(
            start_date.isoformat(),
            end_date.isoformat(),
        )
        return [session.date() for session in sessions]

    # ---- Explicit session-boundary API ---------------------------------

    def pre_market_start(self, value) -> Optional[datetime]:
        d = self._as_date(value)
        return (
            self._combine(d, self.PRE_MARKET_START, self.tz)
            if self.is_trading_day(d)
            else None
        )

    def regular_open(self, value) -> Optional[datetime]:
        d = self._as_date(value)
        return (
            self._combine(d, self.REGULAR_OPEN, self.tz)
            if self.is_trading_day(d)
            else None
        )

    def session_1_end(self, value) -> Optional[datetime]:
        d = self._as_date(value)
        if not self.is_trading_day(d):
            return None
        boundary = (
            self.FRI_SESSION_1_END
            if d.weekday() == 4
            else self.MON_THU_SESSION_1_END
        )
        return self._combine(d, boundary, self.tz)

    def session_2_start(self, value) -> Optional[datetime]:
        d = self._as_date(value)
        if not self.is_trading_day(d):
            return None
        boundary = (
            self.FRI_SESSION_2_START
            if d.weekday() == 4
            else self.MON_THU_SESSION_2_START
        )
        return self._combine(d, boundary, self.tz)

    def regular_close(self, value) -> Optional[datetime]:
        d = self._as_date(value)
        return (
            self._combine(d, self.REGULAR_CLOSE, self.tz)
            if self.is_trading_day(d)
            else None
        )

    def pre_closing_end(self, value) -> Optional[datetime]:
        d = self._as_date(value)
        return (
            self._combine(d, self.PRE_CLOSING_END, self.tz)
            if self.is_trading_day(d)
            else None
        )

    def post_closing_end(self, value) -> Optional[datetime]:
        d = self._as_date(value)
        return (
            self._combine(d, self.POST_CLOSING_END, self.tz)
            if self.is_trading_day(d)
            else None
        )

    def session_at(self, value: datetime) -> MarketSession:
        """Classify a timestamp using IDX local session boundaries."""

        local = self._as_local_datetime(value)
        d = local.date()

        if not self.is_trading_day(d):
            return MarketSession.CLOSED

        pre_market_start = self.pre_market_start(d)
        regular_open = self.regular_open(d)
        session_1_end = self.session_1_end(d)
        session_2_start = self.session_2_start(d)
        regular_close = self.regular_close(d)
        pre_closing_end = self.pre_closing_end(d)
        post_closing_end = self.post_closing_end(d)

        assert pre_market_start is not None
        assert regular_open is not None
        assert session_1_end is not None
        assert session_2_start is not None
        assert regular_close is not None
        assert pre_closing_end is not None
        assert post_closing_end is not None

        if local < pre_market_start:
            return MarketSession.CLOSED
        if local < regular_open:
            return MarketSession.PRE_MARKET
        if local < session_1_end:
            return MarketSession.REGULAR
        if local < session_2_start:
            return MarketSession.LUNCH_BREAK
        if local < regular_close:
            return MarketSession.REGULAR
        if local < pre_closing_end:
            return MarketSession.PRE_CLOSING
        if local <= post_closing_end:
            return MarketSession.POST_CLOSING
        return MarketSession.AFTER_MARKET

    def context_at(self, value: datetime) -> MarketContext:
        """Return market context for the local date of a timestamp."""

        local = self._as_local_datetime(value)
        d = local.date()
        trading_day = self.is_trading_day(d)

        if trading_day:
            session = self.session_at(local)
            pre_market_start = self.pre_market_start(d)
            regular_open = self.regular_open(d)
            session_1_end = self.session_1_end(d)
            session_2_start = self.session_2_start(d)
            regular_close = self.regular_close(d)
            pre_closing_end = self.pre_closing_end(d)
            post_closing_end = self.post_closing_end(d)
            next_day = self.next_trading_day(d)
            previous_day = self.previous_trading_day(d)
        else:
            session = MarketSession.CLOSED
            pre_market_start = None
            regular_open = None
            session_1_end = None
            session_2_start = None
            regular_close = None
            pre_closing_end = None
            post_closing_end = None
            next_day = self._find_next_trading_day(d)
            previous_day = self._find_previous_trading_day(d)

        return MarketContext(
            market=self.MARKET,
            timezone=self.TIMEZONE,
            calendar=self.CALENDAR,
            local_date=d.isoformat(),
            weekday=d.strftime("%A"),
            is_trading_day=trading_day,
            session=session,
            pre_market_start=pre_market_start,
            regular_open=regular_open,
            session_1_end=session_1_end,
            session_2_start=session_2_start,
            regular_close=regular_close,
            pre_closing_end=pre_closing_end,
            post_closing_end=post_closing_end,
            next_trading_day=next_day.isoformat() if next_day else None,
            previous_trading_day=previous_day.isoformat() if previous_day else None,
        )
