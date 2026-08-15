import unittest
from datetime import date, timedelta
from app.final_opportunity.backtest.engine import BacktestObservation, evaluate


class TestBacktestEngine(unittest.TestCase):
    def test_point_in_time_and_metrics(self):
        d = date(2026, 1, 10)
        rows = [
            BacktestObservation("A", d, .9, .10, d - timedelta(days=1)),
            BacktestObservation("B", d, .8, .05, d),
            BacktestObservation("C", d, .2, -.03, d),
            BacktestObservation("D", d, .1, -.05, d),
        ]
        m = evaluate(rows, top_fraction=.5)
        self.assertEqual(m.eligible_observations, 4)
        self.assertEqual(m.hit_rate, .5)
        self.assertAlmostEqual(m.top_bottom_spread, .115)
        self.assertIsNotNone(m.information_coefficient)

    def test_future_feature_is_excluded(self):
        d = date(2026, 1, 10)
        rows = [BacktestObservation("A", d, .9, .10, d + timedelta(days=1))]
        m = evaluate(rows)
        self.assertEqual(m.eligible_observations, 0)
        self.assertIsNone(m.average_forward_return)

    def test_empty_is_safe(self):
        m = evaluate([])
        self.assertEqual(m.observations, 0)
        self.assertIsNone(m.hit_rate)

    def test_bucket_validation(self):
        with self.assertRaises(ValueError):
            evaluate([], top_fraction=0)
