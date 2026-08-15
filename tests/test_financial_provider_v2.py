import unittest

from app.final_opportunity.financial.provider_chain import FinancialProviderChain


class Obs:
    def to_dict(self):
        return {"ticker": "BBRI"}


class GoodIDX:
    name = "idx_public_financial_data"

    def latest_available(self, ticker):
        return Obs()


class EmptyIDX:
    name = "idx_public_financial_data"

    def latest_available(self, ticker):
        return None


class GoodYahoo:
    name = "yahoo_finance_free"

    def fetch(self, ticker):
        return Obs()


class TestFinancialProviderV2(unittest.TestCase):
    def test_idx_is_preferred(self):
        r = FinancialProviderChain(GoodIDX(), GoodYahoo()).fetch("BBRI")
        self.assertEqual(r.provider, "idx_public_financial_data")
        self.assertEqual(r.status, "available")
        self.assertFalse(r.event_time_eligible)

    def test_yahoo_is_explicit_fallback(self):
        r = FinancialProviderChain(EmptyIDX(), GoodYahoo()).fetch("BBRI")
        self.assertEqual(r.provider, "yahoo_finance_free")
        self.assertEqual(r.status, "fallback_available")
        self.assertFalse(r.event_time_eligible)
        self.assertIsNotNone(r.warning)

    def test_no_source(self):
        r = FinancialProviderChain(EmptyIDX(), None).fetch("BBRI")
        self.assertEqual(r.status, "not_found")
        self.assertFalse(r.event_time_eligible)


if __name__ == "__main__":
    unittest.main()
