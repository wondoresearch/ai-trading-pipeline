import unittest

import pandas as pd

from app.price_provider import PriceProvider


class TestPriceProvider(unittest.TestCase):
    def test_symbol_normalization(self):
        provider = PriceProvider()
        self.assertEqual(provider._yf_symbol("BBRI"), "BBRI.JK")
        self.assertEqual(provider._yf_symbol("BBRI.JK"), "BBRI.JK")

    def test_dataframe_normalization(self):
        provider = PriceProvider()
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-08-11", "2026-08-10"]),
                "Close": [4100, 4000],
            }
        )
        normalized = provider._normalize_dataframe(df)
        self.assertEqual(
            normalized["Date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2026-08-10", "2026-08-11"],
        )


if __name__ == "__main__":
    unittest.main()
