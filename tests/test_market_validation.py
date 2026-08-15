import tempfile
import unittest
from datetime import date
from pathlib import Path

from app.final_opportunity.market_validation import validate_file


class TestMarketValidation(unittest.TestCase):
    def write(self, text):
        td = tempfile.TemporaryDirectory()
        path = Path(td.name) / "BBRI.csv"
        path.write_text(text, encoding="utf-8")
        self.addCleanup(td.cleanup)
        return path

    def test_valid(self):
        path = self.write(
            "date,open,high,low,close,volume\n"
            "2026-08-10,100,102,99,101,1000\n"
            "2026-08-11,101,103,100,102,1100\n"
        )
        result = validate_file(path, as_of=date(2026, 8, 13))
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.rows, 2)
        self.assertEqual(result.duplicates, 0)

    def test_duplicate_and_bad_ohlc(self):
        path = self.write(
            "date,open,high,low,close,volume\n"
            "2026-08-10,100,90,99,101,1000\n"
            "2026-08-10,101,103,100,102,1100\n"
        )
        result = validate_file(path, as_of=date(2026, 8, 13))
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.duplicates, 1)
        self.assertEqual(result.invalid_ohlc, 1)

    def test_missing_columns(self):
        path = self.write(
            "date,open,high,low,close\n"
            "2026-08-10,1,2,0,1\n"
        )
        result = validate_file(path, as_of=date(2026, 8, 13))
        self.assertEqual(result.status, "invalid")
        self.assertIn("missing columns", result.message)


if __name__ == "__main__":
    unittest.main()
