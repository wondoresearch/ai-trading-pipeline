import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.market_calendar import IDXMarketCalendar, MarketSession


class TestIDXMarketCalendar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calendar = IDXMarketCalendar()
        cls.tz = ZoneInfo("Asia/Jakarta")

    def dt(self, hour, minute=0, second=0, day=11, month=8, year=2026):
        return datetime(year, month, day, hour, minute, second, tzinfo=self.tz)

    # ---- Trading-day behavior -----------------------------------------

    def test_weekend_is_not_trading_day(self):
        self.assertFalse(
            self.calendar.is_trading_day(datetime(2026, 8, 8).date())
        )

    def test_regular_day_is_trading_day(self):
        self.assertTrue(
            self.calendar.is_trading_day(datetime(2026, 8, 11).date())
        )

    def test_holiday_is_closed(self):
        self.assertFalse(
            self.calendar.is_trading_day(datetime(2026, 8, 17).date())
        )
        self.assertEqual(
            self.calendar.session_at(self.dt(10, day=17)),
            MarketSession.CLOSED,
        )

    def test_next_and_previous_from_non_trading_day(self):
        saturday = datetime(2026, 8, 8).date()
        monday = datetime(2026, 8, 10).date()
        friday = datetime(2026, 8, 7).date()

        self.assertEqual(self.calendar.next_trading_day(saturday), monday)
        self.assertEqual(self.calendar.previous_trading_day(saturday), friday)

    # ---- Session boundary matrix --------------------------------------

    def test_before_pre_market(self):
        self.assertEqual(self.calendar.session_at(self.dt(8, 44, 59)), MarketSession.CLOSED)

    def test_pre_market_start(self):
        self.assertEqual(self.calendar.session_at(self.dt(8, 45)), MarketSession.PRE_MARKET)

    def test_pre_market_before_regular_open(self):
        self.assertEqual(self.calendar.session_at(self.dt(8, 59, 59)), MarketSession.PRE_MARKET)

    def test_regular_open(self):
        self.assertEqual(self.calendar.session_at(self.dt(9, 0)), MarketSession.REGULAR)

    def test_session_one_end(self):
        self.assertEqual(self.calendar.session_at(self.dt(11, 59, 59)), MarketSession.REGULAR)
        self.assertEqual(self.calendar.session_at(self.dt(12, 0)), MarketSession.LUNCH_BREAK)

    def test_session_two_start(self):
        self.assertEqual(self.calendar.session_at(self.dt(13, 29, 59)), MarketSession.LUNCH_BREAK)
        self.assertEqual(self.calendar.session_at(self.dt(13, 30)), MarketSession.REGULAR)

    def test_regular_close_and_pre_closing(self):
        self.assertEqual(self.calendar.session_at(self.dt(15, 49, 59)), MarketSession.REGULAR)
        self.assertEqual(self.calendar.session_at(self.dt(15, 50)), MarketSession.PRE_CLOSING)

    def test_pre_closing_end_and_post_closing(self):
        self.assertEqual(self.calendar.session_at(self.dt(16, 1, 59)), MarketSession.PRE_CLOSING)
        self.assertEqual(self.calendar.session_at(self.dt(16, 2)), MarketSession.POST_CLOSING)

    def test_post_closing_end(self):
        self.assertEqual(self.calendar.session_at(self.dt(16, 15)), MarketSession.POST_CLOSING)
        self.assertEqual(self.calendar.session_at(self.dt(16, 15, 1)), MarketSession.AFTER_MARKET)

    # ---- Friday --------------------------------------------------------

    def test_friday_lunch_break(self):
        friday_1129 = datetime(2026, 8, 14, 11, 29, 59, tzinfo=self.tz)
        friday_1130 = datetime(2026, 8, 14, 11, 30, tzinfo=self.tz)
        friday_1359 = datetime(2026, 8, 14, 13, 59, 59, tzinfo=self.tz)
        friday_1400 = datetime(2026, 8, 14, 14, 0, tzinfo=self.tz)

        self.assertEqual(self.calendar.session_at(friday_1129), MarketSession.REGULAR)
        self.assertEqual(self.calendar.session_at(friday_1130), MarketSession.LUNCH_BREAK)
        self.assertEqual(self.calendar.session_at(friday_1359), MarketSession.LUNCH_BREAK)
        self.assertEqual(self.calendar.session_at(friday_1400), MarketSession.REGULAR)

    # ---- Timezone ------------------------------------------------------

    def test_utc_timestamp_is_converted_to_jakarta(self):
        # 02:00 UTC = 09:00 WIB.
        value = datetime(2026, 8, 11, 2, 0, tzinfo=timezone.utc)
        self.assertEqual(self.calendar.session_at(value), MarketSession.REGULAR)

    def test_naive_datetime_is_explicitly_interpreted_as_jakarta(self):
        value = datetime(2026, 8, 11, 9, 0)
        self.assertEqual(self.calendar.session_at(value), MarketSession.REGULAR)

    # ---- Context / domain contract ------------------------------------

    def test_context_contains_adjacent_sessions(self):
        context = self.calendar.context_at(self.dt(10))
        self.assertEqual(context.session, MarketSession.REGULAR)
        self.assertEqual(context.local_date, "2026-08-11")
        self.assertEqual(context.next_trading_day, "2026-08-12")
        self.assertEqual(context.previous_trading_day, "2026-08-10")

    def test_context_datetime_fields_are_timezone_aware(self):
        context = self.calendar.context_at(self.dt(10))
        self.assertEqual(context.regular_open.hour, 9)
        self.assertEqual(context.regular_open.tzinfo, self.tz)
        self.assertEqual(context.pre_market_start.hour, 8)
        self.assertEqual(context.regular_close.hour, 15)

    def test_context_serialization_is_iso_string(self):
        context = self.calendar.context_at(self.dt(10))
        data = context.to_dict()

        self.assertIsInstance(data["regular_open"], str)
        self.assertEqual(data["regular_open"], "2026-08-11T09:00:00+07:00")
        self.assertEqual(data["session"], "regular")

    def test_closed_day_context_has_no_session_times(self):
        context = self.calendar.context_at(self.dt(10, day=17))
        self.assertFalse(context.is_trading_day)
        self.assertEqual(context.session, MarketSession.CLOSED)
        self.assertIsNone(context.regular_open)
        self.assertEqual(context.next_trading_day, "2026-08-18")
        self.assertEqual(context.previous_trading_day, "2026-08-14")

    # ---- Public boundary methods --------------------------------------

    def test_explicit_boundary_methods(self):
        d = datetime(2026, 8, 11).date()

        self.assertEqual(self.calendar.pre_market_start(d).hour, 8)
        self.assertEqual(self.calendar.regular_open(d).hour, 9)
        self.assertEqual(self.calendar.session_1_end(d).hour, 12)
        self.assertEqual(self.calendar.session_2_start(d).hour, 13)
        self.assertEqual(self.calendar.regular_close(d).hour, 15)
        self.assertEqual(self.calendar.pre_closing_end(d).hour, 16)
        self.assertEqual(self.calendar.post_closing_end(d).hour, 16)


if __name__ == "__main__":
    unittest.main()
