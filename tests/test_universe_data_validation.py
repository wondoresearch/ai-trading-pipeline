import unittest

from app.stock_universe import StockUniverse
from app.universe_data_validation import UniverseDataValidator


class TestUniverseDataValidator(unittest.TestCase):
    def test_missing_and_insufficient_data(self):
        universe = StockUniverse.from_tickers(["AAA", "BBB", "CCC"])

        result = UniverseDataValidator().validate(
            universe,
            historical_returns={
                "AAA": [0.01] * 60,
                "BBB": [0.01] * 20,
            },
            minimum_observations=60,
        )

        self.assertTrue(result[0].eligible)
        self.assertFalse(result[1].eligible)
        self.assertFalse(result[2].eligible)

    def test_eligibility_count(self):
        universe = StockUniverse.from_tickers(["AAA"])
        result = UniverseDataValidator().validate(
            universe,
            {"AAA": [0.01] * 60},
            minimum_observations=60,
        )
        self.assertEqual(result[0].observation_count, 60)
        self.assertEqual(result[0].reason, "eligible")


if __name__ == "__main__":
    unittest.main()
