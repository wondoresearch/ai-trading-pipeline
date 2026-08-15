import csv
import tempfile
import unittest
from pathlib import Path

from app.final_opportunity.market import FreeIDXMarketData, YahooFinanceMarketData
from app.final_opportunity.market_import import import_market_file


class TestFreeIDXMarketData(unittest.TestCase):
    def test_history_reads_normalized_csv(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "BBRI.csv"
            p.write_text(
                "date,open,high,low,close,volume\n"
                "2026-08-10,100,110,90,105,1000\n"
                "2026-08-11,105,115,100,112,1200\n"
            )
            data = FreeIDXMarketData(td)
            rows = data.history("BBRI", "2026-08-10", "2026-08-11")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["close"], 105.0)

    def test_jk_suffix_is_not_required(self):
        self.assertEqual(FreeIDXMarketData.provider_symbol("BBRI.JK"), "BBRI")

    def test_backward_compatible_name_is_local_provider(self):
        self.assertIs(YahooFinanceMarketData, FreeIDXMarketData)


class TestImporter(unittest.TestCase):
    def test_imports_idx_style_columns(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "stock.csv"
            source.write_text(
                "Date,StockCode,OpenPrice,High,Low,Close,Volume\n"
                "2026-08-10,BBRI,100,110,90,105,1000\n"
            )
            out = Path(td) / "out"
            result = import_market_file(str(source), str(out))
            self.assertEqual(result["tickers_written"], 1)
            rows = (out / "BBRI.csv").read_text().splitlines()
            self.assertEqual(rows[1], "2026-08-10,100,110,90,105,1000")


if __name__ == "__main__":
    unittest.main()
