import unittest
from pathlib import Path
from unittest.mock import Mock

from app.final_opportunity.financial.stockanalysis_provider import (
    StockAnalysisFinancialProvider,
)


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "stockanalysis_income.html"


class TestStockAnalysisProvider(unittest.TestCase):
    def provider(self):
        session = Mock()
        response = Mock()
        response.raise_for_status.return_value = None
        response.text = FIXTURE.read_text(encoding="utf-8")
        session.get.return_value = response
        session.headers = {}
        return StockAnalysisFinancialProvider(
            cache_dir="/tmp/stockanalysis-test-cache",
            session=session,
        )

    def test_parses_generic_income_statement(self):
        rows = self.provider().fetch("ANTM")
        self.assertTrue(rows)
        self.assertEqual(rows[0].ticker, "ANTM")
        self.assertEqual(rows[0].period_type, "annual")
        self.assertEqual(rows[0].revenue, 84642439)
        self.assertEqual(rows[0].net_income, 7208834)
        self.assertEqual(rows[0].eps, 299.98)

    def test_provider_is_not_sector_specific(self):
        p = self.provider()
        for ticker in ("BBRI", "ANTM", "ADRO", "ASII"):
            rows = p.fetch(ticker)
            self.assertTrue(rows)

    def test_source_provenance(self):
        row = self.provider().fetch("ANTM")[0]
        self.assertEqual(row.source, "stockanalysis_web")
        self.assertIn("/quote/idx/ANTM/financials/", row.source_url)

    def test_normalizes_jk_suffix(self):
        self.assertIn("/quote/idx/ANTM/financials/", self.provider().url("ANTM.JK"))


if __name__ == "__main__":
    unittest.main()
