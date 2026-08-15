import unittest
from app.final_opportunity.financial.integration_contract import combine_scores

class Adjustment:
    def __init__(self, score, confidence):
        self.score = score
        self.confidence = confidence

class TestContract(unittest.TestCase):
    def test_missing_is_neutral(self):
        r = combine_scores(.42, None)
        self.assertEqual(r.final_score, .42)
        self.assertFalse(r.fundamental_used)

    def test_event_time_guard(self):
        r = combine_scores(.42, Adjustment(.08, 1), event_time_eligible=False)
        self.assertEqual(r.final_score, .42)
        self.assertEqual(r.fundamental_overlay, 0)

    def test_overlay_bounded(self):
        r = combine_scores(.42, Adjustment(99, 1))
        self.assertEqual(r.fundamental_overlay, .10)
        self.assertEqual(r.final_score, .52)

    def test_negative_bounded(self):
        r = combine_scores(.42, Adjustment(-99, 1))
        self.assertEqual(r.fundamental_overlay, -.10)
        self.assertAlmostEqual(r.final_score, .32, places=12)

    def test_confidence_bounded(self):
        self.assertEqual(combine_scores(.4, Adjustment(.05, 3)).confidence, 1)
        self.assertEqual(combine_scores(.4, Adjustment(.05, -3)).confidence, 0)

    def test_deterministic(self):
        a = Adjustment(.07, .8)
        self.assertEqual(combine_scores(.3, a).to_dict(), combine_scores(.3, a).to_dict())

if __name__ == "__main__":
    unittest.main()
