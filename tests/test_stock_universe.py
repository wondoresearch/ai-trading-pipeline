import unittest

from app.stock_universe import StockUniverse


class TestStockUniverse(unittest.TestCase):
    def test_normalizes_tickers(self):
        universe = StockUniverse.from_tickers([" bbri ", "BBCA"])
        self.assertEqual(universe.tickers, ("BBRI", "BBCA"))

    def test_duplicate_is_rejected(self):
        with self.assertRaises(ValueError):
            StockUniverse.from_tickers(["BBRI", "bbri"])

    def test_empty_is_rejected(self):
        with self.assertRaises(ValueError):
            StockUniverse.from_tickers([])

    def test_arbitrary_universe(self):
        universe = StockUniverse.from_tickers(
            ["BBRI", "TLKM", "ASII", "GOTO", "ANTM"]
        )
        self.assertEqual(len(universe), 5)


if __name__ == "__main__":
    unittest.main()
