import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REBUILD = ROOT / "scripts" / "rebuild_pit_observations_safe.py"

class TestPITReconstructionV1(unittest.TestCase):
    def run_builder(self, rows):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "in.json"
            out = td / "out.json"
            src.write_text(json.dumps(rows), encoding="utf-8")
            p = subprocess.run(
                [sys.executable, str(REBUILD),
                 "--input", str(src),
                 "--output", str(out),
                 "--as-of", "2026-08-14"],
                text=True, capture_output=True
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            return json.loads(out.read_text(encoding="utf-8"))

    def base(self, period):
        return {
            "ticker": "BBRI",
            "as_of": "2026-08-14",
            "publication_date": "2026-08-14",
            "financial_period_end": period,
            "price": 3890,
            "fundamental_score": 0.5,
        }

    def test_five_periods_collapse_to_latest_current_snapshot(self):
        rows = [self.base(f"{y}-12-31") for y in range(2021, 2026)]
        p = self.run_builder(rows)
        self.assertEqual(len(p["rows"]), 1)
        self.assertEqual(p["rows"][0]["financial_period_end"], "2025-12-31")
        self.assertFalse(p["rows"][0]["pit_valid"])

    def test_never_invents_historical_publication_date(self):
        rows = [self.base("2024-12-31")]
        p = self.run_builder(rows)
        self.assertFalse(p["historical_pit_ready"])
        self.assertFalse(p["rows"][0]["pit_valid"])

    def test_multiple_tickers_remain_cross_sectional(self):
        a = self.base("2025-12-31")
        b = dict(a, ticker="PTBA", financial_period_end="2025-12-31")
        p = self.run_builder([a, b])
        self.assertEqual({x["ticker"] for x in p["rows"]}, {"BBRI", "PTBA"})

if __name__ == "__main__":
    unittest.main()
