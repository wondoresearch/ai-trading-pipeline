import unittest
from datetime import date
from app.final_opportunity.backtest.pit_v2 import *


class TestPITV2(unittest.TestCase):
    def history(self):
        return [
            FinancialHistory("BBRI", date(2021,12,31), date(2022,3,1), .40, "sa", "listing_timestamp"),
            FinancialHistory("BBRI", date(2022,12,31), date(2023,3,1), .45, "sa", "listing_timestamp"),
            FinancialHistory("BBRI", date(2023,12,31), date(2024,3,1), .50, "sa", "listing_timestamp"),
            FinancialHistory("BBRI", date(2024,12,31), date(2025,3,1), .55, "sa", "listing_timestamp"),
            FinancialHistory("BBRI", date(2025,12,31), date(2026,3,1), .60, "sa", "listing_timestamp"),
        ]

    def test_five_annual_records_become_one_state(self):
        r=build_pit_state("BBRI",date(2026,8,14),price=3890,financial_history=self.history()).to_dict()
        self.assertEqual(r["financial_period_end"],"2025-12-31")
        self.assertEqual(r["fundamental_score"],.60)

    def test_future_publication_rejected(self):
        h=[FinancialHistory("PTBA",date(2025,12,31),date(2026,9,1),.9,"sa","listing_timestamp")]
        r=build_pit_state("PTBA",date(2026,8,14),financial_history=h)
        self.assertIsNone(r.financial_period_end)
        self.assertIsNone(r.fundamental_score)

    def test_unknown_publication_neutral(self):
        h=[FinancialHistory("ADRO",date(2025,12,31),None,.9,"sa")]
        r=build_pit_state("ADRO",date(2026,8,14),financial_history=h)
        self.assertIsNone(r.fundamental_score)

    def test_same_ticker_asof_is_unique(self):
        r=build_pit_state("BBCA",date(2026,8,14),financial_history=self.history())
        a=audit_pit_states([r.to_dict()])
        self.assertEqual(a["duplicate_observations"],0)
        self.assertEqual(a["status"],"PASS")

    def test_cross_sector(self):
        for t in ("BBRI","PTBA","ADRO"):
            r=build_pit_state(t,date(2026,8,14),financial_history=[
                FinancialHistory(t,date(2025,12,31),date(2026,3,1),.7,"sa","listing_timestamp")
            ])
            self.assertEqual(r.ticker,t)
            self.assertEqual(r.fundamental_score,.7)

if __name__=="__main__": unittest.main()
