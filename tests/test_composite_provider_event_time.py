import unittest
from types import SimpleNamespace
from app.final_opportunity.financial.composite_provider import CompositeFinancialProvider

class FakeProvider:
    name="idx_public_financial_data"
    def latest_available(self,ticker): return SimpleNamespace(ticker=ticker, to_dict=lambda:{"ticker":ticker})

class TestCompositeProviderEventTime(unittest.TestCase):
    def test_unverified_publication_is_not_event_time_eligible(self):
        p=CompositeFinancialProvider(idx_provider=FakeProvider(), yahoo_provider=object())
        r=p.status("BBRI")
        self.assertFalse(r["event_time_eligible"])
        self.assertIn("publication timestamp", r["warning"])

if __name__=="__main__": unittest.main()
