import unittest
from app.final_opportunity.financial.models import FinancialObservation
from app.final_opportunity.financial.features import classify_sector, build_features


def obs(sector, sub=""):
    return FinancialObservation(
        ticker="TEST", sector=sector, sub_industry=sub, sub_industry_code="X",
        fs_date="2025-12-31", fiscal_year_end="Dec", statement_type="A",
        auditor_opinion="-", assets=1000, liabilities=400, equity=600,
        sales=1200, ebt=200, profit=150, profit_attributed=140, eps=10,
        book_value=60, pe=12, pbv=2, debt_equity=0.67, roa=5, roe=20, npm=12.5
    )


class TestFinancialFeatures(unittest.TestCase):
    def test_bank_is_financial(self):
        self.assertEqual(classify_sector("Financials", "Banks"), "financial")

    def test_mining_is_resource(self):
        self.assertEqual(classify_sector("Energy", "Coal Production"), "resource")

    def test_manufacturer_is_industrial(self):
        self.assertEqual(classify_sector("Industrials", "Machinery"), "industrial")

    def test_feature_set_is_sector_aware(self):
        result = build_features([obs("Financials", "Banks")])
        self.assertEqual(result.sector_group, "financial")
        self.assertGreaterEqual(result.quality, 0)
        self.assertLessEqual(result.financial_score, 1)


if __name__ == "__main__":
    unittest.main()
