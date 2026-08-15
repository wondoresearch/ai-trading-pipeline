import unittest
from types import SimpleNamespace
from app.final_opportunity.integration.orchestrator import OpportunityOrchestrator

class TestOpportunityOrchestrator(unittest.TestCase):
    def test_default_event_guard_fails_closed(self):
        o=OpportunityOrchestrator(market_score=lambda _: .5,fundamental_adjustment=lambda _: SimpleNamespace(score=.1,confidence=.8))
        r=o.evaluate("BBRI.JK")
        self.assertFalse(r.event_time_eligible); self.assertEqual(r.fundamental_overlay,0.0); self.assertEqual(r.composite_score,.5)

    def test_positive_overlay_requires_explicit_eligibility(self):
        o=OpportunityOrchestrator(market_score=lambda _: .5,fundamental_adjustment=lambda _: SimpleNamespace(score=.1,confidence=.8),event_time_eligible=lambda _: True)
        r=o.evaluate("BBRI.JK")
        self.assertAlmostEqual(r.fundamental_overlay,.08); self.assertAlmostEqual(r.composite_score,.58)

    def test_overlay_is_bounded_before_confidence(self):
        o=OpportunityOrchestrator(market_score=lambda _: .95,fundamental_adjustment=lambda _: SimpleNamespace(score=9.,confidence=1.),event_time_eligible=lambda _: True)
        r=o.evaluate("PTBA"); self.assertAlmostEqual(r.fundamental_overlay,.10); self.assertEqual(r.composite_score,1.0)

    def test_ranking_is_deterministic(self):
        vals={"A":.4,"B":.7,"C":.5}; o=OpportunityOrchestrator(market_score=lambda t: vals[t],fundamental_adjustment=lambda _: None)
        self.assertEqual([r.ticker for r in o.rank(["A","C","B"])],["B","C","A"])

if __name__=="__main__": unittest.main()
