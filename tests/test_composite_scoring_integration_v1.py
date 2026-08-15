import unittest

from app.final_opportunity.composite_scoring import integrate_fundamentals


class Adjustment:
    def __init__(self, score, confidence):
        self.score = score
        self.confidence = confidence


class TestCompositeScoringIntegration(unittest.TestCase):
    def test_missing_fundamentals_keep_base_score(self):
        r = integrate_fundamentals("BBRI", 0.42, None)
        self.assertEqual(r.base_score, 0.42)
        self.assertEqual(r.final_score, 0.42)
        self.assertEqual(r.fundamental_overlay, 0.0)
        self.assertFalse(r.fundamental_used)

    def test_event_time_guard_keeps_base_score(self):
        r = integrate_fundamentals(
            "BBRI", 0.42, Adjustment(0.08, 1.0),
            event_time_eligible=False,
        )
        self.assertEqual(r.final_score, 0.42)
        self.assertEqual(r.fundamental_overlay, 0.0)
        self.assertFalse(r.fundamental_used)

    def test_positive_overlay_is_bounded(self):
        r = integrate_fundamentals("PTBA", 0.42, Adjustment(99, 1.0))
        self.assertAlmostEqual(r.fundamental_overlay, 0.10, places=12)
        self.assertAlmostEqual(r.final_score, 0.52, places=12)

    def test_negative_overlay_is_bounded(self):
        r = integrate_fundamentals("ADRO", 0.42, Adjustment(-99, 1.0))
        self.assertAlmostEqual(r.fundamental_overlay, -0.10, places=12)
        self.assertAlmostEqual(r.final_score, 0.32, places=12)

    def test_confidence_is_bounded(self):
        high = integrate_fundamentals("BBCA", 0.4, Adjustment(0.05, 99))
        low = integrate_fundamentals("BBCA", 0.4, Adjustment(0.05, -99))
        self.assertEqual(high.fundamental_confidence, 1.0)
        self.assertEqual(low.fundamental_confidence, 0.0)

    def test_base_score_is_not_overwritten(self):
        r = integrate_fundamentals("PTBA", 0.73, Adjustment(0.04, 0.5))
        self.assertEqual(r.base_score, 0.73)
        self.assertNotEqual(r.final_score, r.base_score)

    def test_sector_agnostic_ticker_contract(self):
        for ticker in ("BBRI", "BBCA", "PTBA", "ADRO"):
            r = integrate_fundamentals(ticker, 0.5, Adjustment(0.04, 0.8))
            self.assertEqual(r.ticker, ticker)
            self.assertTrue(r.fundamental_used)

    def test_serialization(self):
        r = integrate_fundamentals("BBRI", 0.5, Adjustment(0.04, 0.8))
        d = r.to_dict()
        self.assertEqual(d["ticker"], "BBRI")
        self.assertIn("final_score", d)
        self.assertIn("fundamental_overlay", d)


if __name__ == "__main__":
    unittest.main()
