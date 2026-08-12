import unittest

from app.risk_model import RiskEstimator


class TestRiskEstimator(unittest.TestCase):
    def test_metrics_are_positive(self):
        returns = [
            0.01, -0.005, 0.008, 0.002, -0.003,
            0.006, -0.004, 0.005, 0.003, -0.002,
            0.004, 0.001, -0.003, 0.007, -0.001,
            0.002, 0.004, -0.005, 0.003, 0.002,
        ]

        result = RiskEstimator().estimate(
            returns,
            minimum_observations=20,
        )

        self.assertGreater(result.volatility, 0.0)
        self.assertGreaterEqual(result.downside_deviation, 0.0)
        self.assertGreaterEqual(result.max_drawdown, 0.0)
        self.assertEqual(result.observation_count, 20)

    def test_insufficient_history(self):
        with self.assertRaises(ValueError):
            RiskEstimator().estimate(
                [0.01] * 10,
                minimum_observations=20,
            )


if __name__ == "__main__":
    unittest.main()
