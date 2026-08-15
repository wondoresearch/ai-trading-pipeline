import unittest
from types import SimpleNamespace

from app.final_opportunity.integration.orchestrator import OpportunityOrchestrator


class TestOpportunityOrchestrator(unittest.TestCase):
    def test_positive_overlay_is_confidence_weighted(self):
        o = OpportunityOrchestrator(
            market_score=lambda _: 0.50,
            fundamental_adjustment=lambda _: SimpleNamespace(score=0.10, confidence=0.80),
        )
        r = o.evaluate("BBRI.JK")
        self.assertEqual(r.ticker, "BBRI")
        self.assertAlmostEqual(r.fundamental_overlay, 0.08)
        self.assertAlmostEqual(r.composite_score, 0.58)

    def test_overlay_is_bounded_before_confidence(self):
        o = OpportunityOrchestrator(
            market_score=lambda _: 0.95,
            fundamental_adjustment=lambda _: SimpleNamespace(score=9.0, confidence=1.0),
        )
        r = o.evaluate("PTBA")
        self.assertAlmostEqual(r.fundamental_overlay, 0.10)
        self.assertEqual(r.composite_score, 1.0)

    def test_event_guard_forces_neutral(self):
        o = OpportunityOrchestrator(
            market_score=lambda _: 0.50,
            fundamental_adjustment=lambda _: SimpleNamespace(score=0.10, confidence=1.0),
            event_time_eligible=lambda _: False,
        )
        r = o.evaluate("BBCA")
        self.assertFalse(r.event_time_eligible)
        self.assertEqual(r.fundamental_overlay, 0.0)
        self.assertEqual(r.composite_score, 0.50)

    def test_missing_fundamental_is_neutral(self):
        o = OpportunityOrchestrator(
            market_score=lambda _: 0.42,
            fundamental_adjustment=lambda _: None,
        )
        r = o.evaluate("ADRO")
        self.assertEqual(r.fundamental_overlay, 0.0)
        self.assertEqual(r.composite_score, 0.42)

    def test_confidence_clamped(self):
        o = OpportunityOrchestrator(
            market_score=lambda _: 0.40,
            fundamental_adjustment=lambda _: SimpleNamespace(score=0.10, confidence=99),
        )
        r = o.evaluate("PTBA")
        self.assertEqual(r.fundamental_confidence, 1.0)
        self.assertEqual(r.composite_score, 0.50)

    def test_negative_overlay_cannot_make_score_negative(self):
        o = OpportunityOrchestrator(
            market_score=lambda _: 0.02,
            fundamental_adjustment=lambda _: SimpleNamespace(score=-9.0, confidence=1),
        )
        r = o.evaluate("X")
        self.assertEqual(r.fundamental_overlay, -0.10)
        self.assertEqual(r.composite_score, 0.0)

    def test_ranking_is_deterministic(self):
        vals = {"A": .4, "B": .7, "C": .5}
        o = OpportunityOrchestrator(
            market_score=lambda t: vals[t],
            fundamental_adjustment=lambda _: None,
        )
        self.assertEqual([r.ticker for r in o.rank(["A", "C", "B"])], ["B", "C", "A"])

    def test_serializable(self):
        o = OpportunityOrchestrator(
            market_score=lambda _: .5,
            fundamental_adjustment=lambda _: None,
        )
        d = o.evaluate("BBRI").to_dict()
        self.assertEqual(d["ticker"], "BBRI")
        self.assertIn("composite_score", d)
        self.assertIsInstance(d["reasons"], list)


if __name__ == "__main__":
    unittest.main()
