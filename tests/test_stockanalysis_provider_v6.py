import unittest
from pathlib import Path
from unittest.mock import Mock

from app.final_opportunity.financial.stockanalysis_provider import (
    StockAnalysisFinancialProvider,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "stockanalysis_income_real_shape.html"


class TestStockAnalysisProviderV6(unittest.TestCase):
    def provider(self):
        session = Mock()
        response = Mock()
        response.raise_for_status.return_value = None
        response.text = FIXTURE.read_text(encoding="utf-8")
        session.get.return_value = response
        session.headers = {}
        return StockAnalysisFinancialProvider(
            cache_dir="/tmp/stockanalysis-v6-test-cache",
            session=session,
        )

    def test_parses_real_world_stockanalysis_shape(self):
        rows = self.provider().fetch("BBRI")
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(rows[0].ticker, "BBRI")
        self.assertEqual(rows[0].period, "FY 2025")
        self.assertEqual(rows[0].period_end, "2025-12-31")
        self.assertEqual(rows[0].revenue, 140208835)
        self.assertEqual(rows[0].net_income, 56652384)
        self.assertEqual(rows[0].eps, 376.0)

    def test_sector_agnostic(self):
        # The parser only consumes generic accounting lines; it does not
        # assume bank/coal/manufacturing-specific metrics.
        rows = self.provider().fetch("PTBA")
        self.assertTrue(rows)

    def test_provenance(self):
        row = self.provider().fetch("BBRI")[0]
        self.assertEqual(row.source, "stockanalysis_web")
        self.assertIn("/quote/idx/BBRI/financials/", row.source_url)
        self.assertEqual(row.currency, "IDR")
        self.assertEqual(row.unit, "millions")

    def test_jk_suffix(self):
        self.assertIn("/quote/idx/BBRI/financials/", self.provider().url("BBRI.JK"))


if __name__ == "__main__":
    unittest.main()
