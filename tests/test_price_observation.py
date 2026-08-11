import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from app.event_time import EventTimeResolution, EventTimeRule
from app.market_calendar import MarketSession
from app.price_observation import (
    HistoricalPriceObservationEngine,
    ObservationHorizon,
    PriceObservationStatus,
)


class FakeProvider:
    def __init__(self, frame=None, error=None):
        self.frame = frame
        self.error = error
        self.calls = []

    def get_history(self, ticker, start, end):
        self.calls.append((ticker, start, end))
        if self.error:
            raise self.error
        return self.frame.copy()


class FakeCalendar:
    def __init__(self):
        self.tz = ZoneInfo("Asia/Jakarta")

    def is_trading_day(self, value):
        return value.weekday() < 5

    def previous_trading_day(self, value):
        d = value
        while True:
            from datetime import timedelta
            d -= timedelta(days=1)
            if self.is_trading_day(d):
                return d

    def next_trading_day(self, value):
        d = value
        while True:
            from datetime import timedelta
            d += timedelta(days=1)
            if self.is_trading_day(d):
                return d

    def regular_close(self, value):
        return datetime.combine(value, datetime.min.time().replace(hour=15, minute=50), tzinfo=self.tz)


def price_frame(dates):
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Open": [100 + i for i in range(len(dates))],
            "High": [101 + i for i in range(len(dates))],
            "Low": [99 + i for i in range(len(dates))],
            "Close": [1000 + 10 * i for i in range(len(dates))],
            "Adj Close": [1000 + 10 * i for i in range(len(dates))],
            "Volume": [10000 + i for i in range(len(dates))],
        }
    )


class TestHistoricalPriceObservationEngine(unittest.TestCase):
    def setUp(self):
        self.tz = ZoneInfo("Asia/Jakarta")
        self.calendar = FakeCalendar()
        self.resolution = EventTimeResolution(
            event_id="evt-1",
            ticker="BBCA",
            event_time=datetime(2026, 8, 11, 10, 5, tzinfo=self.tz),
            market_date=datetime(2026, 8, 11, tzinfo=self.tz).date(),
            market_session=MarketSession.REGULAR,
            effective_time=datetime(2026, 8, 11, 10, 5, tzinfo=self.tz),
            resolution_rule=EventTimeRule.REGULAR_SESSION,
            is_trading_day=True,
            is_tradeable_at_event=True,
            is_same_session_effective=True,
        )

    def test_baseline_is_previous_trading_session(self):
        dates = [
            "2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13",
            "2026-08-14", "2026-08-17", "2026-08-18", "2026-08-19",
            "2026-08-20", "2026-08-21", "2026-08-24", "2026-08-25",
        ]
        provider = FakeProvider(price_frame(dates))
        result = HistoricalPriceObservationEngine(provider, self.calendar).resolve(self.resolution)
        self.assertEqual(result.baseline.baseline_date.isoformat(), "2026-08-10")
        self.assertEqual(result.baseline.price, 1000.0)
        self.assertEqual(result.observations[0].horizon, ObservationHorizon.EVENT_DAY)
        self.assertEqual(result.observations[0].observation_date.isoformat(), "2026-08-11")
        self.assertEqual(result.observations[1].observation_date.isoformat(), "2026-08-12")
        self.assertEqual(result.observations[-1].observation_date.isoformat(), "2026-08-25")
        self.assertEqual(result.status, PriceObservationStatus.OBSERVED)

    def test_after_market_resolution_uses_next_day_as_event_day_and_previous_close_as_baseline(self):
        resolution = EventTimeResolution(
            event_id="evt-2",
            ticker="BBCA",
            event_time=datetime(2026, 8, 11, 18, 0, tzinfo=self.tz),
            market_date=datetime(2026, 8, 11, tzinfo=self.tz).date(),
            market_session=MarketSession.AFTER_MARKET,
            effective_time=datetime(2026, 8, 12, 9, 0, tzinfo=self.tz),
            resolution_rule=EventTimeRule.AFTER_MARKET_TO_NEXT_OPEN,
            is_trading_day=True,
            is_tradeable_at_event=False,
            is_same_session_effective=False,
        )
        dates = [
            "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14",
            "2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20",
            "2026-08-21", "2026-08-24", "2026-08-25", "2026-08-26",
        ]
        provider = FakeProvider(price_frame(dates))
        result = HistoricalPriceObservationEngine(provider, self.calendar).resolve(resolution)
        self.assertEqual(result.baseline.baseline_date.isoformat(), "2026-08-11")
        self.assertEqual(result.observations[0].observation_date.isoformat(), "2026-08-12")

    def test_missing_baseline_is_explicit(self):
        dates = ["2026-08-11", "2026-08-12", "2026-08-13"]
        provider = FakeProvider(price_frame(dates))
        result = HistoricalPriceObservationEngine(provider, self.calendar).resolve(self.resolution)
        self.assertEqual(result.baseline.status, PriceObservationStatus.MISSING_BASELINE)
        self.assertEqual(result.status, PriceObservationStatus.MISSING_BASELINE)

    def test_missing_forward_data_is_not_treated_as_missing_baseline(self):
        dates = ["2026-08-10", "2026-08-11", "2026-08-12"]
        provider = FakeProvider(price_frame(dates))
        result = HistoricalPriceObservationEngine(provider, self.calendar).resolve(self.resolution)
        self.assertEqual(result.baseline.status, PriceObservationStatus.OBSERVED)
        self.assertEqual(result.observations[0].status, PriceObservationStatus.OBSERVED)
        self.assertEqual(result.observations[-1].status, PriceObservationStatus.MISSING_OBSERVATION)
        self.assertEqual(result.status, PriceObservationStatus.INSUFFICIENT_FORWARD_DATA)

    def test_provider_exception_is_explicit(self):
        provider = FakeProvider(error=RuntimeError("provider unavailable"))
        result = HistoricalPriceObservationEngine(provider, self.calendar).resolve(self.resolution)
        self.assertEqual(result.status, PriceObservationStatus.PROVIDER_ERROR)
        self.assertEqual(result.baseline.status, PriceObservationStatus.PROVIDER_ERROR)
        self.assertTrue(all(item.status == PriceObservationStatus.PROVIDER_ERROR for item in result.observations))

    def test_duplicate_dates_are_deterministically_deduplicated(self):
        frame = price_frame(["2026-08-10", "2026-08-10", "2026-08-11"])
        frame.loc[1, "Adj Close"] = 9999
        provider = FakeProvider(frame)
        result = HistoricalPriceObservationEngine(provider, self.calendar).resolve(self.resolution)
        self.assertEqual(result.baseline.price, 9999.0)

    def test_invalid_adjusted_close_is_explicit(self):
        frame = price_frame(["2026-08-10", "2026-08-11"])
        frame.loc[0, "Adj Close"] = None
        provider = FakeProvider(frame)
        result = HistoricalPriceObservationEngine(provider, self.calendar).resolve(self.resolution)
        self.assertEqual(result.baseline.status, PriceObservationStatus.INVALID_DATA)
        self.assertEqual(result.status, PriceObservationStatus.INVALID_DATA)

    def test_invalid_schema_is_explicit(self):
        frame = pd.DataFrame({"Date": pd.to_datetime(["2026-08-10", "2026-08-11"]), "Close": [1000, 1010]})
        provider = FakeProvider(frame)
        result = HistoricalPriceObservationEngine(provider, self.calendar).resolve(self.resolution)
        self.assertEqual(result.status, PriceObservationStatus.INVALID_DATA)
        self.assertEqual(result.baseline.status, PriceObservationStatus.INVALID_DATA)

    def test_provider_window_is_trading_calendar_aware(self):
        dates = [
            "2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13",
            "2026-08-14", "2026-08-17", "2026-08-18", "2026-08-19",
            "2026-08-20", "2026-08-21", "2026-08-24", "2026-08-25",
        ]
        provider = FakeProvider(price_frame(dates))
        HistoricalPriceObservationEngine(provider, self.calendar).resolve(self.resolution)
        self.assertEqual(provider.calls[0][1], "2026-08-10")
        self.assertEqual(provider.calls[0][2], "2026-08-26")


if __name__ == "__main__":
    unittest.main()
