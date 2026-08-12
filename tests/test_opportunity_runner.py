
import json
import tempfile
import unittest
from pathlib import Path

from app.opportunity_config import OpportunityRunnerConfig
from app.opportunity_runner import load_tickers, run


class TestOpportunityRunner(unittest.TestCase):
    def test_load_tickers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "universe.txt"
            path.write_text("# comment\n bbri \nBBCA\n\n", encoding="utf-8")
            self.assertEqual(load_tickers(path), ["bbri", "BBCA"])

    def test_run_writes_report(self):
        returns = [
            0.01, -0.005, 0.008, 0.002, -0.003,
            0.006, -0.004, 0.005, 0.003, -0.002,
            0.004, 0.001, -0.003, 0.007, -0.001,
            0.002, 0.004, -0.005, 0.003, 0.002,
        ] * 3

        prepared = {
            "predictions": {"AAA": 0.08, "BBB": 0.04},
            "confidence": {"AAA": 0.8, "BBB": 0.7},
            "historical_returns": {
                "AAA": returns,
                "BBB": returns,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "ranking.json"
            result = run(
                ["AAA", "BBB"],
                prepared,
                OpportunityRunnerConfig(
                    output_path=output,
                    minimum_history=60,
                    risk_minimum_history=20,
                    top_n=2,
                ),
            )
            self.assertEqual(len(result.ranking), 2)
            self.assertTrue(output.exists())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["ranking"][0]["ticker"], "AAA")
            self.assertFalse(payload["live_trading"])


if __name__ == "__main__":
    unittest.main()
