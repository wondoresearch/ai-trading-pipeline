import unittest

from app.opportunity_score import OpportunityScorer


class TestOpportunityScorer(unittest.TestCase):
    def test_higher_return_can_improve_score(self):
        scorer = OpportunityScorer()

        low = scorer.score(
            expected_return=0.04,
            confidence=0.8,
            volatility=0.10,
            downside_deviation=0.08,
            max_drawdown=0.12,
        )
        high = scorer.score(
            expected_return=0.08,
            confidence=0.8,
            volatility=0.10,
            downside_deviation=0.08,
            max_drawdown=0.12,
        )

        self.assertGreater(high.score, low.score)

    def test_invalid_confidence(self):
        with self.assertRaises(ValueError):
            OpportunityScorer().score(
                expected_return=0.05,
                confidence=1.5,
                volatility=0.10,
                downside_deviation=0.05,
                max_drawdown=0.10,
            )


if __name__ == "__main__":
    unittest.main()
