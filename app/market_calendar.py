from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Optional

from zoneinfo import ZoneInfo

import exchange_calendars as xcals


class MarketSession(str, Enum):
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    LUNCH_BREAK = "lunch_break"
    AFTER_MARKET = "after_market"


@dataclass(frozen=True)
class MarketContext:
    market: str
    timezone: str
    calendar: str
    local_date: str
    is_trading_day: bool
    session: MarketSession
    market_open: Optional[str]
    break_start: Optional[str]
    break_end: Optional[str]
    market_close: Optional[str]
    next_trading_day: Optional[str]
    previous_trading_day: Optional[str]

    def to_dict(self):
        return {
            "market": self.market,
            "timezone": self.timezone,
            "calendar": self.calendar,
            "local_date": self.local_date,
            "is_trading_day": self.is_trading_day,
            "session": self.session.value,
            "market_open": self.market_open,
            "break_start": self.break_start,
            "break_end": self.break_end,
            "market_close": self.market_close,
            "next_trading_day": self.next_trading_day,
            "previous_trading_day": self.previous_trading_day,
        }


class IDXMarketCalendar:
    """Market calendar for Indonesia Stock Exchange (IDX).

    The actual trading-day/holiday schedule comes from exchange_calendars'
    XIDX calendar. This avoids maintaining a hand-written holiday list.

    Times returned by the public API are ISO-8601 timestamps in Asia/Jakarta.
    """

    MARKET = "IDX"
    CALENDAR = "XIDX"
    TIMEZONE = "Asia/Jakarta"

    # IDX regular session starts at 09:00 local time. The exchange calendar
    # supplies the authoritative open/break/close timestamps.
    PRE_MARKET_START = time(8, 45)

    def __init__(self):
        self.tz = ZoneInfo(self.TIMEZONE)
        self.calendar = xcals.get_calendar(self.CALENDAR)

    @staticmethod
    def _as_date(value) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return datetime.fromisoformat(str(value)).date()

    def _schedule_row(self, session_date: date):
        session = session_date.isoformat()
        try:
            schedule = self.calendar.schedule.loc[session:session]
        except Exception:
            return None

        if schedule.empty:
            return None

        return schedule.iloc[0]

    @staticmethod
    def _iso(value):
        if value is None:
            return None
        return value.isoformat()

    def is_trading_day(self, value) -> bool:
        session_date = self._as_date(value)
        return bool(self.calendar.is_session(session_date.isoformat()))

    def next_trading_day(self, value) -> date:
        session_date = self._as_date(value)
        return self.calendar.next_session(session_date.isoformat()).date()

    def previous_trading_day(self, value) -> date:
        session_date = self._as_date(value)
        return self.calendar.previous_session(session_date.isoformat()).date()

    def trading_days_between(self, start, end):
        start_date = self._as_date(start)
        end_date = self._as_date(end)
        sessions = self.calendar.sessions_in_range(
            start_date.isoformat(),
            end_date.isoformat(),
        )
        return [session.date() for session in sessions]

    def market_open(self, value) -> Optional[datetime]:
        row = self._schedule_row(self._as_date(value))
        if row is None:
            return None
        return row["market_open"].tz_convert(self.tz).to_pydatetime()

    def market_close(self, value) -> Optional[datetime]:
        row = self._schedule_row(self._as_date(value))
        if row is None:
            return None
        return row["market_close"].tz_convert(self.tz).to_pydatetime()

    def break_start(self, value) -> Optional[datetime]:
        row = self._schedule_row(self._as_date(value))
        if row is None:
            return None
        value = row.get("break_start")
        if value is None or str(value) == "NaT":
            return None
        return value.tz_convert(self.tz).to_pydatetime()

    def break_end(self, value) -> Optional[datetime]:
        row = self._schedule_row(self._as_date(value))
        if row is None:
            return None
        value = row.get("break_end")
        if value is None or str(value) == "NaT":
            return None
        return value.tz_convert(self.tz).to_pydatetime()

    def session_at(self, value) -> MarketSession:
        """Classify a Jakarta-local or timezone-aware datetime.

        For a non-trading day the result is CLOSED.
        On a trading day:
          - before 08:45: CLOSED
          - 08:45 until open: PRE_MARKET
          - open to break: REGULAR
          - break to break_end: LUNCH_BREAK
          - break_end to close: REGULAR
          - after close: AFTER_MARKET
        """
        if not isinstance(value, datetime):
            raise TypeError("session_at() requires a datetime")

        if value.tzinfo is None:
            value = value.replace(tzinfo=self.tz)
        else:
            value = value.astimezone(self.tz)

        session_date = value.date()
        row = self._schedule_row(session_date)

        if row is None:
            return MarketSession.CLOSED

        market_open = self.market_open(session_date)
        market_close = self.market_close(session_date)
        break_start = self.break_start(session_date)
        break_end = self.break_end(session_date)

        pre_market = datetime.combine(
            session_date,
            self.PRE_MARKET_START,
            tzinfo=self.tz,
        )

        if value < pre_market:
            return MarketSession.CLOSED
        if value < market_open:
            return MarketSession.PRE_MARKET
        if break_start and break_end and break_start <= value < break_end:
            return MarketSession.LUNCH_BREAK
        if value < market_close:
            return MarketSession.REGULAR
        return MarketSession.AFTER_MARKET

    def context_at(self, value) -> MarketContext:
        """Return a complete market context for an event timestamp."""
        if not isinstance(value, datetime):
            raise TypeError("context_at() requires a datetime")

        local = (
            value.replace(tzinfo=self.tz)
            if value.tzinfo is None
            else value.astimezone(self.tz)
        )

        trading_day = self.is_trading_day(local.date())

        if trading_day:
            next_day = self.next_trading_day(local.date())
            previous_day = self.previous_trading_day(local.date())
            open_dt = self.market_open(local.date())
            break_start_dt = self.break_start(local.date())
            break_end_dt = self.break_end(local.date())
            close_dt = self.market_close(local.date())
            session = self.session_at(local)
        else:
            # For weekends/holidays, expose the surrounding trading sessions.
            # The next/previous methods require a valid session, so search
            # backwards/forwards from the nearest dates.
            next_day = self._find_next_trading_day(local.date())
            previous_day = self._find_previous_trading_day(local.date())
            open_dt = break_start_dt = break_end_dt = close_dt = None
            session = MarketSession.CLOSED

        return MarketContext(
            market=self.MARKET,
            timezone=self.TIMEZONE,
            calendar=self.CALENDAR,
            local_date=local.date().isoformat(),
            is_trading_day=trading_day,
            session=session,
            market_open=self._iso(open_dt),
            break_start=self._iso(break_start_dt),
            break_end=self._iso(break_end_dt),
            market_close=self._iso(close_dt),
            next_trading_day=next_day.isoformat() if next_day else None,
            previous_trading_day=previous_day.isoformat() if previous_day else None,
        )

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
