import unittest
from unittest.mock import patch, Mock
from io import StringIO
from datetime import date
import pandas as pd

from app.final_opportunity.financial.idx_provider import IDXFinancialDataProvider
from app.final_opportunity.financial.resolver import FinancialResolver

class TestFinancialProviderV3(unittest.TestCase):
    def test_detects_data_unavailable(self):
        html = "<table><tr><td>Data tidak tersedia</td></tr></table>"
        with patch("app.final_opportunity.financial.idx_provider.requests.get") as g:
            g.return_value = Mock(status_code=200, text=html)
            self.assertEqual(IDXFinancialDataProvider().fetch_month(2025,9,["BBRI"]), [])

    def test_parses_normal_table(self):
        df = pd.DataFrame({
            "Sector":["Energy"], "Sub Industry Code":["A1"],
            "Sub Industry":["Coal Production"], "Code":["BBRI"],
            "FS Date":["2025-09-30"], "Assets, b.IDR":["100"],
            "Liabilities, b.IDR":["40"], "Equity, b.IDR":["60"],
            "Sales, b.IDR":["80"], "EBT, b.IDR":["20"],
            "Profit for the Period":["15"], "Profit attr.to owner's":["15"],
            "EPS, IDR":["10"], "Book Value, IDR":["20"],
            "P/E Ratio, x":["2"], "Price to BV, x":["1"],
            "D/E Ratio, x":["0.67"], "ROA, %":["15"], "ROE, %":["25"],
            "NPM, %":["18.75"],
        })
        html = df.to_html(index=False)
        with patch("app.final_opportunity.financial.idx_provider.requests.get") as g:
            g.return_value = Mock(status_code=200, text=html)
            rows = IDXFinancialDataProvider().fetch_month(2025,10,["BBRI"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].sector, "Energy")
            self.assertEqual(rows[0].sub_industry, "Coal Production")

    def test_non_bank_sector_supported(self):
        self.test_parses_normal_table()

    def test_resolver_not_found(self):
        provider = Mock()
        provider.name = "test"
        provider.latest_available.return_value = None
        r = FinancialResolver(provider).resolve("BBRI")
        self.assertEqual(r.status, "not_found")
        self.assertFalse(r.event_time_eligible)

    def test_resolver_event_time(self):
        provider = Mock()
        provider.name = "test"
        provider.latest_available.return_value = Mock(
            fs_date="2026-08-01",
            to_dict=lambda: {"ticker":"BBRI"}
        )
        r = FinancialResolver(provider).resolve("BBRI", date(2026,8,13))
        self.assertEqual(r.status, "available")
        self.assertTrue(r.event_time_eligible)

if __name__ == "__main__":
    unittest.main()
