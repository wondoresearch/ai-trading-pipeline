
import json
import math
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from app.opportunity_report import build_report, write_report
from app.opportunity_pipeline import OpportunityPipelineResult
from app.stock_universe import StockUniverse
from app.universe_data_validation import UniverseDataStatus


class TestOpportunityReport(unittest.TestCase):
    def _result(self):
        return OpportunityPipelineResult(
            universe=StockUniverse.from_tickers(["AAA"]),
            data_status=(
                UniverseDataStatus("AAA", True, 60, "eligible"),
            ),
            ranking=(),
        )

    def test_json_safe_and_deterministic(self):
        result = self._result()
        first = build_report(result)
        second = build_report(result)
        self.assertEqual(first, second)
        serialized = json.dumps(first, allow_nan=False)
        self.assertIn('"live_trading": false', serialized)

    def test_non_finite_becomes_none(self):
        from app.opportunity_report import _safe
        self.assertIsNone(_safe(float("nan")))
        self.assertIsNone(_safe(float("inf")))
        self.assertEqual(_safe(datetime(2026, 1, 2)), "2026-01-02T00:00:00")
        self.assertEqual(_safe(date(2026, 1, 2)), "2026-01-02")

    def test_write_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            write_report(self._result(), path)
            self.assertTrue(path.exists())
            parsed = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["schema_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
