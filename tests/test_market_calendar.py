import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from app.market_calendar import IDXMarketCalendar, MarketSession


class TestIDXMarketCalendar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calendar = IDXMarketCalendar()
        cls.tz = ZoneInfo("Asia/Jakarta")

    def test_weekend_is_not_trading_day(self):
        self.assertFalse(
            self.calendar.is_trading_day(
                datetime(2026, 8, 8).date()
            )
        )

    def test_regular_day_is_trading_day(self):
        self.assertTrue(
            self.calendar.is_trading_day(
                datetime(2026, 8, 11).date()
            )
        )

    def test_pre_market(self):
        dt = datetime(2026, 8, 11, 8, 50, tzinfo=self.tz)
        self.assertEqual(
            self.calendar.session_at(dt),
            MarketSession.PRE_MARKET,
        )

    def test_regular_market(self):
        dt = datetime(2026, 8, 11, 10, 0, tzinfo=self.tz)
        self.assertEqual(
            self.calendar.session_at(dt),
            MarketSession.REGULAR,
        )

    def test_lunch_break(self):
        dt = datetime(2026, 8, 11, 12, 30, tzinfo=self.tz)
        self.assertEqual(
            self.calendar.session_at(dt),
            MarketSession.LUNCH_BREAK,
        )

    def test_after_market(self):
        dt = datetime(2026, 8, 11, 17, 0, tzinfo=self.tz)
        self.assertEqual(
            self.calendar.session_at(dt),
            MarketSession.AFTER_MARKET,
        )

    def test_holiday_is_closed(self):
        # 17 August is Indonesia's Independence Day and is a standard IDX
        # closure in the XIDX exchange calendar.
        dt = datetime(2026, 8, 17, 10, 0, tzinfo=self.tz)
        self.assertEqual(
            self.calendar.session_at(dt),
            MarketSession.CLOSED,
        )

    def test_context_contains_adjacent_sessions(self):
        dt = datetime(2026, 8, 11, 10, 0, tzinfo=self.tz)
        context = self.calendar.context_at(dt)

        self.assertTrue(context.is_trading_day)
        self.assertEqual(context.session, MarketSession.REGULAR)
        self.assertEqual(context.next_trading_day, "2026-08-12")
        self.assertEqual(context.previous_trading_day, "2026-08-10")


if __name__ == "__main__":
    unittest.main()
