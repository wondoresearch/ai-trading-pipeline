import tempfile
import unittest
from datetime import date
from pathlib import Path

from app.final_opportunity.market_provider import MarketDataProvider
from app.final_opportunity.provider_status import check_provider


class FakeProvider(MarketDataProvider):
    name = "fake"

    def __init__(self, data):
        self.data = data

    def history(self, ticker, start, end):
        return self.data.get(ticker, [])


class TestProviderStatus(unittest.TestCase):
    def rows(self, d):
        return [{"date": d, "open": 1, "high": 2, "low": 1, "close": 2, "volume": 1}]

    def test_healthy_and_fresh(self):
        p = FakeProvider({"BBRI": self.rows("2026-08-12")})
        r = check_provider(p, ["BBRI"], 3, date(2026, 8, 13))
        self.assertEqual(r.status, "healthy")
        self.assertTrue(r.fresh)
        self.assertEqual(r.age_days, 1)

    def test_stale(self):
        p = FakeProvider({"BBRI": self.rows("2026-08-01")})
        r = check_provider(p, ["BBRI"], 3, date(2026, 8, 13))
        self.assertEqual(r.status, "stale")
        self.assertFalse(r.fresh)
        self.assertEqual(r.age_days, 12)
        self.assertEqual(r.stale_tickers, ["BBRI"])

    def test_mixed_fresh_and_stale_is_stale(self):
        p = FakeProvider({
            "BBRI": self.rows("2026-08-12"),
            "BBCA": self.rows("2026-08-01"),
        })
        r = check_provider(p, ["BBRI", "BBCA"], 3, date(2026, 8, 13))
        self.assertEqual(r.status, "stale")
        self.assertFalse(r.fresh)
        self.assertEqual(r.stale_tickers, ["BBCA"])

    def test_missing_is_degraded(self):
        p = FakeProvider({"BBRI": self.rows("2026-08-12")})
        r = check_provider(p, ["BBRI", "BBCA"], 3, date(2026, 8, 13))
        self.assertEqual(r.status, "degraded")
        self.assertFalse(r.fresh)
        self.assertEqual(r.missing_tickers, ["BBCA"])

    def test_unavailable(self):
        r = check_provider(FakeProvider({}), ["BBRI"], 3, date(2026, 8, 13))
        self.assertEqual(r.status, "unavailable")
        self.assertFalse(r.available)

    def test_empty(self):
        r = check_provider(FakeProvider({}), [], 3, date(2026, 8, 13))
        self.assertEqual(r.status, "unavailable")

    def test_boundary_is_fresh(self):
        p = FakeProvider({"BBRI": self.rows("2026-08-10")})
        r = check_provider(p, ["BBRI"], 3, date(2026, 8, 13))
        self.assertEqual(r.status, "healthy")

    def test_one_day_over_boundary_is_stale(self):
        p = FakeProvider({"BBRI": self.rows("2026-08-09")})
        r = check_provider(p, ["BBRI"], 3, date(2026, 8, 13))
        self.assertEqual(r.status, "stale")
