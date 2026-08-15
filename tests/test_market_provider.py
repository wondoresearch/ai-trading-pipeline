import unittest
import tempfile
from pathlib import Path

from app.final_opportunity.market import FreeIDXMarketData
from app.final_opportunity.market_provider import MarketDataProvider


class TestMarketDataProvider(unittest.TestCase):

    def test_free_idx_provider_implements_abstraction(self):
        provider = FreeIDXMarketData(tempfile.mkdtemp())
        self.assertIsInstance(provider, MarketDataProvider)
        self.assertEqual(provider.name, "local_idx")

    def test_health(self):
        provider = FreeIDXMarketData(tempfile.mkdtemp())
        result = provider.health()
        self.assertEqual(result["provider"], "local_idx")
        self.assertTrue(result["available"])

    def test_latest(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "BBRI.csv").write_text(
                "date,open,high,low,close,volume\n"
                "2026-08-10,100,102,99,101,1000\n"
                "2026-08-11,101,103,100,102,1100\n",
                encoding="utf-8",
            )

            provider = FreeIDXMarketData(td)
            result = provider.latest("BBRI")

            self.assertEqual(result["date"], "2026-08-11")
            self.assertEqual(result["close"], 102.0)


if __name__ == "__main__":
    unittest.main()
