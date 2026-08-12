import unittest

from app.opportunity_pipeline import OpportunityPipeline


class TestOpportunityPipeline(unittest.TestCase):
    def test_pipeline_filters_ineligible_stocks(self):
        returns = [
            0.01, -0.005, 0.008, 0.002, -0.003,
            0.006, -0.004, 0.005, 0.003, -0.002,
            0.004, 0.001, -0.003, 0.007, -0.001,
            0.002, 0.004, -0.005, 0.003, 0.002,
        ]

        history = {
            "AAA": returns * 3,
            "BBB": returns,
        }

        result = OpportunityPipeline().run(
            tickers=["AAA", "BBB"],
            predictions={"AAA": 0.06, "BBB": 0.04},
            confidence={"AAA": 0.8, "BBB": 0.7},
            historical_returns=history,
            data_minimum_observations=60,
            risk_minimum_observations=20,
        )

        self.assertEqual(len(result.ranking), 1)
        self.assertEqual(result.ranking[0].ticker, "AAA")
        self.assertFalse(result.data_status[1].eligible)


if __name__ == "__main__":
    unittest.main()
