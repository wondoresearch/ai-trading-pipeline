import json
import tempfile
import unittest
from datetime import date

from app.final_opportunity.backtest.point_in_time import (
    PITObservation,
    build_point_in_time_rows,
    composite_score,
    forward_return,
    rank_cross_section,
)


class TestPITBacktest(unittest.TestCase):
    def test_future_publication_is_rejected(self):
        obs = PITObservation("BBRI", date(2026, 3, 31), date(2026, 4, 2), 4000, .8, .8, .8)
        self.assertFalse(obs.point_in_time_valid)
        self.assertEqual(build_point_in_time_rows([obs]), [])

    def test_same_day_publication_is_allowed(self):
        obs = PITObservation("BBRI", date(2026, 3, 31), date(2026, 3, 31), 4000, .8, .8, .8)
        rows = build_point_in_time_rows([obs])
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["composite_score"], .8)

    def test_missing_fundamental_is_renormalized(self):
        obs = PITObservation("PTBA", date(2026, 3, 31), date(2026, 3, 1), 3000, .6, .8, None)
        rows = build_point_in_time_rows([obs])
        self.assertAlmostEqual(rows[0]["composite_score"], (0.8*.40 + 0.6*.35)/.75)

    def test_score_bounded(self):
        obs = PITObservation("ADRO", date(2026, 3, 31), date(2026, 3, 1), 1000, 9, -3, 4)
        self.assertEqual(composite_score(obs), 0.60)

    def test_forward_return_uses_future_only_as_label(self):
        r = forward_return({"BBRI": 100}, "BBRI", {"BBRI": 110})
        self.assertAlmostEqual(r, .10)

    def test_forward_return_invalid_price(self):
        self.assertIsNone(forward_return({"BBRI": 0}, "BBRI", {"BBRI": 110}))

    def test_deterministic_ranking(self):
        rows = [
            {"ticker": "BBCA", "composite_score": .8},
            {"ticker": "BBRI", "composite_score": .8},
            {"ticker": "PTBA", "composite_score": .7},
        ]
        ranked = rank_cross_section(rows)
        self.assertEqual([x["ticker"] for x in ranked], ["BBCA", "BBRI", "PTBA"])
        self.assertEqual([x["rank"] for x in ranked], [1, 2, 3])

    def test_normalizes_ticker(self):
        obs = PITObservation("BBRI.JK", date(2026, 3, 31), date(2026, 3, 1), 4000, .7, .7, .7)
        self.assertEqual(build_point_in_time_rows([obs])[0]["ticker"], "BBRI")


if __name__ == "__main__":
    unittest.main()
