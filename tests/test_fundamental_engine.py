import unittest
from types import SimpleNamespace

from app.final_opportunity.financial.fundamental_engine import (
    build_fundamental_features,
    _sector_profile,
)


def row(**kw):
    base = dict(
        ticker="TEST", sector="Industrials", sub_industry="Manufacturing",
        revenue=1000, gross_profit=300, operating_income=150,
        net_income=100, eps=10, roe=0.10, debt_equity=0.5, pbv=1.0
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestFundamentalEngine(unittest.TestCase):
    def test_growth_and_margin(self):
        current = row(revenue=1200, net_income=150, eps=15)
        previous = row(revenue=1000, net_income=100, eps=10)
        f = build_fundamental_features(current, previous)
        self.assertAlmostEqual(f.revenue_growth, .20)
        self.assertAlmostEqual(f.net_income_growth, .50)
        self.assertAlmostEqual(f.net_margin, .125)
        self.assertGreater(f.financial_score, .5)
        self.assertEqual(f.trend, "improving")

    def test_missing_metrics_are_not_zero(self):
        current = row(gross_profit=None, operating_income=None)
        previous = row(gross_profit=None, operating_income=None)
        f = build_fundamental_features(current, previous)
        self.assertIsNotNone(f)
        self.assertGreaterEqual(f.confidence, .45)

    def test_banking_profile(self):
        self.assertEqual(_sector_profile("Financials", "Banks"), "banking")
        f = build_fundamental_features(
            row(sector="Financials", sub_industry="Banks", roe=.15),
            row(sector="Financials", sub_industry="Banks", roe=.12),
        )
        self.assertEqual(f.sector_profile, "banking")

    def test_resources_profile(self):
        self.assertEqual(_sector_profile("Energy", "Coal Mining"), "resources")

    def test_industrial_profile(self):
        self.assertEqual(_sector_profile("Industrials", "Manufacturing"), "industrial")


if __name__ == "__main__":
    unittest.main()
