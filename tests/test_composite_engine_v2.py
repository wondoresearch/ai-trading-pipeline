import unittest
from types import SimpleNamespace
from app.final_opportunity.financial.composite_engine import combine

class TestCompositeEngineV2(unittest.TestCase):
    def f(self, score=0.05, confidence=0.9):
        return SimpleNamespace(score=score, confidence=confidence, feature_set=object())

    def test_missing_is_neutral(self):
        r = combine(0.20, None)
        self.assertEqual(r.score, 0.20)
        self.assertFalse(r.fundamental_used)

    def test_event_time_guard_is_neutral(self):
        r = combine(0.20, self.f(0.10, 1.0), event_time_eligible=False)
        self.assertEqual(r.score, 0.20)
        self.assertFalse(r.fundamental_used)

    def test_overlay_is_bounded(self):
        self.assertAlmostEqual(combine(0.20, self.f(9.0, 1.0)).score, 0.30)
        self.assertAlmostEqual(combine(0.20, self.f(-9.0, 1.0)).score, 0.10)

    def test_confidence_is_clamped(self):
        self.assertAlmostEqual(combine(0.20, self.f(0.10, 9.0)).score, 0.30)
        self.assertAlmostEqual(combine(0.20, self.f(0.10, -9.0)).score, 0.20)

    def test_confidence_scales_overlay(self):
        self.assertAlmostEqual(combine(0.20, self.f(0.10, 0.5)).score, 0.25)

    def test_serializable(self):
        d = combine(0.20, self.f(0.10, 0.8)).to_dict()
        self.assertEqual(d["score"], 0.28)
        self.assertIn("fundamental_used", d)

if __name__ == "__main__":
    unittest.main()
