import unittest
from types import SimpleNamespace
from app.final_opportunity.integration.research_engine import DataStatus, ResearchOpportunityEngine


class TestResearchEngine(unittest.TestCase):
    def test_healthy_eligible(self):
        e = ResearchOpportunityEngine(
            market_score=lambda _: .6,
            fundamental_adjustment=lambda _: SimpleNamespace(score=.1, confidence=.8),
            event_time_eligible=lambda _: True,
            data_status=lambda _: DataStatus("healthy", "healthy", "healthy"),
        )
        r = e.evaluate("BBRI.JK")
        self.assertTrue(r.eligible)
        self.assertAlmostEqual(r.result.composite_score, .68)

    def test_event_guard_blocks(self):
        e = ResearchOpportunityEngine(
            market_score=lambda _: .6,
            fundamental_adjustment=lambda _: SimpleNamespace(score=.1, confidence=1),
            event_time_eligible=lambda _: False,
            data_status=lambda _: DataStatus("healthy", "healthy", "healthy"),
        )
        r = e.evaluate("BBCA")
        self.assertFalse(r.eligible)
        self.assertEqual(r.eligibility_reason, "event_time_not_eligible")
        self.assertEqual(r.result.fundamental_overlay, 0)

    def test_market_unavailable_blocks(self):
        e = ResearchOpportunityEngine(
            market_score=lambda _: .6,
            fundamental_adjustment=lambda _: None,
            event_time_eligible=lambda _: True,
            data_status=lambda _: DataStatus("unavailable", "healthy", "healthy"),
        )
        r = e.evaluate("PTBA")
        self.assertFalse(r.eligible)
        self.assertEqual(r.eligibility_reason, "market_unavailable")

    def test_financial_unavailable_can_be_neutral_when_missing(self):
        e = ResearchOpportunityEngine(
            market_score=lambda _: .6,
            fundamental_adjustment=lambda _: None,
            event_time_eligible=lambda _: True,
            data_status=lambda _: DataStatus("healthy", "healthy", "unavailable"),
        )
        r = e.evaluate("ADRO")
        self.assertTrue(r.eligible)
        self.assertEqual(r.result.status, "no_fundamentals")

    def test_deterministic_rank(self):
        vals = {"B": .8, "A": .8, "C": .2}
        e = ResearchOpportunityEngine(
            market_score=lambda t: vals[t],
            fundamental_adjustment=lambda _: None,
            event_time_eligible=lambda _: True,
        )
        self.assertEqual([x.result.ticker for x in e.rank(["C", "B", "A"])], ["A", "B", "C"])
