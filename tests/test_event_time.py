import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from app.event_time import EventTimeInput, EventTimeRule, IDXEventTimeEngine
from app.market_calendar import MarketSession


class TestIDXEventTimeEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = IDXEventTimeEngine()
        cls.tz = ZoneInfo("Asia/Jakarta")

    def event(self, dt: datetime) -> EventTimeInput:
        return EventTimeInput("evt-1", "BBCA", dt)

    def test_rejects_naive_datetime(self):
        with self.assertRaises(ValueError):
            self.event(datetime(2026, 8, 11, 10, 5))

    def test_regular_session(self):
        result = self.engine.resolve(
            self.event(datetime(2026, 8, 11, 10, 5, tzinfo=self.tz))
        )
        self.assertEqual(result.market_session, MarketSession.REGULAR)
        self.assertEqual(result.effective_time.hour, 10)
        self.assertEqual(result.resolution_rule, EventTimeRule.REGULAR_SESSION)
        self.assertTrue(result.is_tradeable_at_event)
        self.assertTrue(result.is_same_session_effective)

    def test_pre_market_maps_to_regular_open(self):
        result = self.engine.resolve(
            self.event(datetime(2026, 8, 11, 8, 50, tzinfo=self.tz))
        )
        self.assertEqual(result.market_session, MarketSession.PRE_MARKET)
        self.assertEqual(result.effective_time, datetime(2026, 8, 11, 9, 0, tzinfo=self.tz))
        self.assertEqual(result.resolution_rule, EventTimeRule.PRE_MARKET_TO_OPEN)
        self.assertFalse(result.is_tradeable_at_event)

    def test_before_pre_market_maps_to_next_trading_day(self):
        result = self.engine.resolve(
            self.event(datetime(2026, 8, 11, 8, 30, tzinfo=self.tz))
        )
        self.assertEqual(result.market_session, MarketSession.CLOSED)
        self.assertEqual(result.effective_time, datetime(2026, 8, 12, 9, 0, tzinfo=self.tz))
        self.assertEqual(
            result.resolution_rule,
            EventTimeRule.CLOSED_BEFORE_PRE_MARKET_TO_NEXT_OPEN,
        )

    def test_lunch_break_monday_to_thursday(self):
        result = self.engine.resolve(
            self.event(datetime(2026, 8, 11, 12, 30, tzinfo=self.tz))
        )
        self.assertEqual(result.market_session, MarketSession.LUNCH_BREAK)
        self.assertEqual(result.effective_time, datetime(2026, 8, 11, 13, 30, tzinfo=self.tz))
        self.assertEqual(result.resolution_rule, EventTimeRule.LUNCH_BREAK_TO_SESSION_II)

    def test_friday_lunch_maps_to_1400(self):
        result = self.engine.resolve(
            self.event(datetime(2026, 8, 14, 12, 30, tzinfo=self.tz))
        )
        self.assertEqual(result.market_session, MarketSession.LUNCH_BREAK)
        self.assertEqual(result.effective_time, datetime(2026, 8, 14, 14, 0, tzinfo=self.tz))

    def test_pre_closing_is_same_day(self):
        result = self.engine.resolve(
            self.event(datetime(2026, 8, 11, 15, 55, tzinfo=self.tz))
        )
        self.assertEqual(result.market_session, MarketSession.PRE_CLOSING)
        self.assertEqual(result.effective_time, datetime(2026, 8, 11, 15, 55, tzinfo=self.tz))
        self.assertEqual(result.resolution_rule, EventTimeRule.PRE_CLOSING)
        self.assertTrue(result.is_tradeable_at_event)

    def test_post_closing_maps_to_next_open(self):
        result = self.engine.resolve(
            self.event(datetime(2026, 8, 11, 16, 5, tzinfo=self.tz))
        )
        self.assertEqual(result.market_session, MarketSession.POST_CLOSING)
        self.assertEqual(result.effective_time, datetime(2026, 8, 12, 9, 0, tzinfo=self.tz))
        self.assertFalse(result.is_same_session_effective)

    def test_after_market_maps_to_next_open(self):
        result = self.engine.resolve(
            self.event(datetime(2026, 8, 11, 18, 0, tzinfo=self.tz))
        )
        self.assertEqual(result.market_session, MarketSession.AFTER_MARKET)
        self.assertEqual(result.effective_time, datetime(2026, 8, 12, 9, 0, tzinfo=self.tz))
        self.assertEqual(result.resolution_rule, EventTimeRule.AFTER_MARKET_TO_NEXT_OPEN)

    def test_weekend_maps_to_next_trading_day(self):
        result = self.engine.resolve(
            self.event(datetime(2026, 8, 15, 10, 0, tzinfo=self.tz))
        )
        self.assertEqual(result.market_session, MarketSession.CLOSED)
        self.assertEqual(result.effective_time, datetime(2026, 8, 18, 9, 0, tzinfo=self.tz))
        self.assertEqual(result.resolution_rule, EventTimeRule.NON_TRADING_DAY_TO_NEXT_OPEN)

    def test_utc_timestamp_is_converted_to_jakarta(self):
        result = self.engine.resolve(
            self.event(datetime(2026, 8, 11, 3, 5, tzinfo=ZoneInfo("UTC")))
        )
        self.assertEqual(result.event_time, datetime(2026, 8, 11, 10, 5, tzinfo=self.tz))
        self.assertEqual(result.market_session, MarketSession.REGULAR)

    def test_resolution_serializes_datetimes_as_iso_strings(self):
        result = self.engine.resolve(
            self.event(datetime(2026, 8, 11, 10, 5, tzinfo=self.tz))
        )
        payload = result.to_dict()
        self.assertIsInstance(payload["event_time"], str)
        self.assertIsInstance(payload["effective_time"], str)
        self.assertTrue(payload["event_time"].endswith("+07:00"))
        self.assertTrue(payload["effective_time"].endswith("+07:00"))

    def test_effective_time_never_precedes_event_time(self):
        cases = [
            datetime(2026, 8, 11, 8, 30, tzinfo=self.tz),
            datetime(2026, 8, 11, 8, 50, tzinfo=self.tz),
            datetime(2026, 8, 11, 10, 5, tzinfo=self.tz),
            datetime(2026, 8, 11, 12, 30, tzinfo=self.tz),
            datetime(2026, 8, 11, 15, 55, tzinfo=self.tz),
            datetime(2026, 8, 11, 16, 5, tzinfo=self.tz),
            datetime(2026, 8, 11, 18, 0, tzinfo=self.tz),
            datetime(2026, 8, 15, 10, 0, tzinfo=self.tz),
        ]
        for dt in cases:
            with self.subTest(dt=dt):
                result = self.engine.resolve(self.event(dt))
                self.assertLessEqual(result.event_time, result.effective_time)

    def test_resolution_from_news_event_uses_published_at_utc(self):
        from app.event_schema import NewsEvent

        event = NewsEvent(
            event_id="evt-news-1",
            news_id="news-1",
            ticker="BBCA",
            company="Bank Central Asia",
            title="Example",
            summary="Example",
            url="https://example.com/news",
            source="test",
            published_at="2026-08-11T10:05:00+07:00",
            published_at_utc="2026-08-11T03:05:00+00:00",
            published_timezone="Asia/Jakarta",
            sentiment="positive",
            sentiment_score=0.9,
            signed_score=0.9,
            entity_confidence=1.0,
            matched_alias="BBCA",
        )
        result = self.engine.resolve_news_event(event)
        self.assertEqual(result.event_time, datetime(2026, 8, 11, 10, 5, tzinfo=self.tz))

    def test_resolution_from_news_event_rejects_missing_timestamp(self):
        from app.event_schema import NewsEvent

        event = NewsEvent(
            event_id="evt-news-2",
            news_id="news-2",
            ticker="BBCA",
            company=None,
            title="Example",
            summary="Example",
            url="https://example.com/news",
            source="test",
            published_at=None,
            published_at_utc=None,
            published_timezone="Asia/Jakarta",
            sentiment=None,
            sentiment_score=None,
            signed_score=0.0,
            entity_confidence=None,
            matched_alias=None,
        )
        with self.assertRaises(ValueError):
            self.engine.resolve_news_event(event)


if __name__ == "__main__":
    unittest.main()
