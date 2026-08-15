import unittest
from datetime import date

from app.final_opportunity.market_provider import MarketDataProvider
from app.final_opportunity.market_readiness import MarketReadinessGate, MarketReadinessError


class FakeProvider(MarketDataProvider):
    name = "fake"

    def __init__(self, data):
        self.data = data

    def history(self, ticker, start, end):
        return self.data.get(ticker, [])

def row(d):
    return [{"date": d, "open": 1, "high": 2, "low": 1, "close": 2, "volume": 1}]


class TestMarketReadiness(unittest.TestCase):
    def test_ready(self):
        p = FakeProvider({"BBRI": row("2026-08-12")})
        r = MarketReadinessGate(3).evaluate(p, ["BBRI"], date(2026, 8, 13))
        self.assertTrue(r.ready)
        self.assertEqual(r.status, "healthy")

    def test_stale_blocked(self):
        p = FakeProvider({"BBRI": row("2026-08-01")})
        r = MarketReadinessGate(3).evaluate(p, ["BBRI"], date(2026, 8, 13))
        self.assertFalse(r.ready)
        self.assertEqual(r.status, "stale")

    def test_missing_blocked(self):
        p = FakeProvider({})
        r = MarketReadinessGate(3).evaluate(p, ["BBRI"], date(2026, 8, 13))
        self.assertFalse(r.ready)
        self.assertEqual(r.status, "unavailable")

    def test_empty_blocked(self):
        p = FakeProvider({})
        r = MarketReadinessGate(3).evaluate(p, [], date(2026, 8, 13))
        self.assertFalse(r.ready)

    def test_require_ready_raises(self):
        p = FakeProvider({"BBRI": row("2026-08-01")})
        with self.assertRaises(MarketReadinessError):
            MarketReadinessGate(3).require_ready(
                p, ["BBRI"], date(2026, 8, 13)
            )
