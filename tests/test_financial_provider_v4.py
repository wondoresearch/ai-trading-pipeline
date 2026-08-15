import sys
import types
import unittest
from unittest.mock import patch
import pandas as pd

from app.final_opportunity.financial.yahoo_provider import YahooFinanceFinancialProvider


class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol
        self.income_stmt = pd.DataFrame(
            {
                pd.Timestamp("2025-12-31"): [1000.0, 100.0, 120.0, 1.2]
            },
            index=["Total Revenue", "Pretax Income", "Net Income", "Diluted EPS"],
        )
        self.quarterly_income_stmt = self.income_stmt
        self.balance_sheet = pd.DataFrame(
            {pd.Timestamp("2025-12-31"): [5000.0, 4000.0, 1000.0]},
            index=[
                "Total Assets",
                "Total Liabilities Net Minority Interest",
                "Stockholders Equity",
            ],
        )
        self.quarterly_balance_sheet = self.balance_sheet

    @property
    def info(self):
        return {"sector": "Financial Services", "industry": "Banks - Regional"}


class TestYahooFinancialProvider(unittest.TestCase):
    def test_normalizes_statement(self):
        fake = types.SimpleNamespace(Ticker=FakeTicker)
        with patch.dict(sys.modules, {"yfinance": fake}):
            obs = YahooFinanceFinancialProvider("/tmp/test-yahoo").fetch("BBRI")
        self.assertIsNotNone(obs)
        self.assertEqual(obs.ticker, "BBRI")
        self.assertEqual(obs.fs_date, "2025-12-31")
        self.assertEqual(obs.sales, 1000.0)
        self.assertEqual(obs.profit, 120.0)
        self.assertAlmostEqual(obs.roe, 0.12, places=6)
        self.assertEqual(obs.sector, "Financial Services")

    def test_jk_symbol_is_not_duplicated(self):
        fake = types.SimpleNamespace(Ticker=FakeTicker)
        with patch.dict(sys.modules, {"yfinance": fake}):
            obs = YahooFinanceFinancialProvider("/tmp/test-yahoo").fetch("BBRI.JK")
        self.assertEqual(obs.ticker, "BBRI")


if __name__ == "__main__":
    unittest.main()
