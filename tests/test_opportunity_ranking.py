import unittest

from app.opportunity_ranking import OpportunityRanker
from app.stock_universe import StockUniverse


class TestOpportunityRanker(unittest.TestCase):
    def test_ranks_user_universe(self):
        universe = StockUniverse.from_tickers(["AAA", "BBB", "CCC"])

        returns = [
            0.01, -0.005, 0.008, 0.002, -0.003,
            0.006, -0.004, 0.005, 0.003, -0.002,
            0.004, 0.001, -0.003, 0.007, -0.001,
            0.002, 0.004, -0.005, 0.003, 0.002,
        ]

        history = {
            "AAA": returns,
            "BBB": [x * 1.5 for x in returns],
            "CCC": [x * 0.7 for x in returns],
        }

        result = OpportunityRanker().rank(
            universe=universe,
            predictions={"AAA": 0.05, "BBB": 0.08, "CCC": 0.03},
            confidence={"AAA": 0.8, "BBB": 0.6, "CCC": 0.9},
            historical_returns=history,
            minimum_observations=20,
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(
            {item.ticker for item in result},
            {"AAA", "BBB", "CCC"},
        )
        self.assertEqual(
            [item.rank for item in result],
            [1, 2, 3],
        )

    def test_missing_prediction_is_rejected(self):
        universe = StockUniverse.from_tickers(["AAA"])

        with self.assertRaises(ValueError):
            OpportunityRanker().rank(
                universe=universe,
                predictions={},
                confidence={"AAA": 0.8},
                historical_returns={"AAA": [0.01] * 20},
                minimum_observations=20,
            )


if __name__ == "__main__":
    unittest.main()
