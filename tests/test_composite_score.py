import unittest
from types import SimpleNamespace

from app.final_opportunity.financial.composite_score import combine_scores
from app.final_opportunity.financial.enrichment import FundamentalAdjustment


class TestCompositeScoreContract(unittest.TestCase):
    """Contract tests for the composite layer.

    Important: the test intentionally does not construct FinancialFeatureSet
    directly. Its constructor is an existing internal contract owned by the
    financial feature module; this layer only requires a non-None feature_set.
    """

    def feature_set(self, financial_score=0.8, quality=0.9):
        return SimpleNamespace(
            financial_score=financial_score,
            quality=quality,
            to_dict=lambda: {
                "financial_score": financial_score,
                "quality": quality,
            },
        )

    def test_positive_fundamental_overlay_is_bounded(self):
        f = FundamentalAdjustment(0.50, 0.9, self.feature_set())
        r = combine_scores(0.20, f)
        self.assertAlmostEqual(r.fundamental_adjustment, 0.10)
        self.assertAlmostEqual(r.score, 0.30)
        self.assertTrue(r.fundamental_used)
        self.assertEqual(r.reason, "sector_aware_fundamental_overlay")

    def test_missing_fundamentals_are_neutral(self):
        r = combine_scores(0.20, None)
        self.assertAlmostEqual(r.score, 0.20)
        self.assertAlmostEqual(r.fundamental_adjustment, 0.0)
        self.assertFalse(r.fundamental_used)
        self.assertEqual(r.reason, "fundamental_unavailable")

    def test_event_time_guard_is_neutral(self):
        f = FundamentalAdjustment(0.08, 1.0, self.feature_set())
        r = combine_scores(0.20, f, event_time_eligible=False)
        self.assertAlmostEqual(r.score, 0.20)
        self.assertFalse(r.fundamental_used)
        self.assertEqual(r.reason, "fundamental_not_event_time_eligible")

    def test_confidence_is_bounded(self):
        f = FundamentalAdjustment(-0.05, 3.0, self.feature_set())
        r = combine_scores(0.20, f)
        self.assertAlmostEqual(r.confidence, 1.0)

    def test_negative_overlay_is_bounded(self):
        f = FundamentalAdjustment(-0.50, 0.9, self.feature_set())
        r = combine_scores(0.20, f)
        self.assertAlmostEqual(r.fundamental_adjustment, -0.10)
        self.assertAlmostEqual(r.score, 0.10)

    def test_custom_bound_is_respected(self):
        f = FundamentalAdjustment(0.50, 0.9, self.feature_set())
        r = combine_scores(0.20, f, max_adjustment=0.03)
        self.assertAlmostEqual(r.fundamental_adjustment, 0.03)
        self.assertAlmostEqual(r.score, 0.23)


if __name__ == "__main__":
    unittest.main()
