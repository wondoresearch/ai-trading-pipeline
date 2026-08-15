import tempfile
import unittest
from pathlib import Path

from app.final_opportunity.market import (
    FreeIDXMarketData,
    YahooFinanceMarketData,
)


class TestMarket(unittest.TestCase):
    def test_provider_symbol_mapping(self):
        self.assertEqual(FreeIDXMarketData.provider_symbol("BBRI"), "BBRI")
        self.assertEqual(FreeIDXMarketData.provider_symbol("BBRI.JK"), "BBRI")

    def test_backward_compatible_name_is_local_provider(self):
        self.assertIs(YahooFinanceMarketData, FreeIDXMarketData)

    def test_history_normalization(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "BBRI.csv"
            path.write_text(
                "date,open,high,low,close,volume\n"
                "2026-08-10,100,102,99,101,1000\n"
                "2026-08-11,101,103,100,102,1100\n",
                encoding="utf-8",
            )

            rows = FreeIDXMarketData(td).history(
                "BBRI",
                "2026-08-01",
                "2026-08-12",
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["date"], "2026-08-10")
            self.assertEqual(rows[0]["open"], 100.0)
            self.assertEqual(rows[0]["high"], 102.0)
            self.assertEqual(rows[0]["low"], 99.0)
            self.assertEqual(rows[0]["close"], 101.0)
            self.assertEqual(rows[0]["volume"], 1000.0)

    def test_history_requires_local_data(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                FreeIDXMarketData(td).history(
                    "BBRI",
                    "2026-08-01",
                    "2026-08-12",
                )


if __name__ == "__main__":
    unittest.main()
