import tempfile
import unittest
from pathlib import Path

from app.final_opportunity.historical_ingest import import_source


class TestHistoricalIngest(unittest.TestCase):
    def test_idx_style_import_and_dedup(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "raw.csv"
            out = Path(td) / "out"
            src.write_text(
                "Date,StockCode,OpenPrice,High,Low,Close,Volume\n"
                "2026-08-10,BBRI,100,110,90,105,1000\n"
                "2026-08-10,BBRI,100,110,90,105,1000\n"
                "2026-08-11,BBRI,105,115,100,112,1200\n"
                "2026-08-11,BBCA,9000,9100,8900,9050,2000\n",
                encoding="utf-8",
            )
            report = import_source(str(src), str(out))
            self.assertEqual(report["tickers_written"], 2)
            rows = (out / "BBRI.csv").read_text().splitlines()
            self.assertEqual(len(rows), 3)
            self.assertTrue((out / "_provenance.json").exists())

    def test_rejects_bad_ohlc(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "raw.csv"
            out = Path(td) / "out"
            src.write_text(
                "Date,StockCode,OpenPrice,High,Low,Close,Volume\n"
                "2026-08-10,BBRI,120,110,90,105,1000\n",
                encoding="utf-8",
            )
            report = import_source(str(src), str(out))
            self.assertEqual(report["rejected_rows"], 1)
            self.assertFalse((out / "BBRI.csv").exists())


if __name__ == "__main__":
    unittest.main()
