import unittest

from app.cross_sectional_evaluation import CrossSectionalEvaluator


class TestCrossSectionalEvaluator(unittest.TestCase):

    def test_top_k_and_excess_return(self):
        evaluator = CrossSectionalEvaluator()

        metrics = evaluator.evaluate(
            predicted_returns=[0.10, 0.08, 0.03, -0.02],
            realized_returns=[0.06, 0.04, 0.01, -0.03],
            benchmark_returns=[0.02, 0.02, 0.02, 0.02],
            top_k=2,
        )

        self.assertAlmostEqual(
            metrics.top_k_mean_return,
            0.05,
        )

        self.assertAlmostEqual(
            metrics.top_k_mean_excess_return,
            0.03,
        )

    def test_rank_ic_positive(self):
        evaluator = CrossSectionalEvaluator()

        metrics = evaluator.evaluate(
            predicted_returns=[0.10, 0.08, 0.03, -0.02],
            realized_returns=[0.06, 0.04, 0.01, -0.03],
            benchmark_returns=[0.02, 0.02, 0.02, 0.02],
            top_k=2,
        )

        self.assertGreater(
            metrics.rank_ic,
            0.9,
        )

    def test_hit_rate(self):
        evaluator = CrossSectionalEvaluator()

        metrics = evaluator.evaluate(
            predicted_returns=[0.10, 0.08, 0.03, -0.02],
            realized_returns=[0.06, -0.04, 0.01, -0.03],
            benchmark_returns=[0.02, 0.02, 0.02, 0.02],
            top_k=2,
        )

        self.assertAlmostEqual(
            metrics.hit_rate,
            0.5,
        )

    def test_top_k_cannot_exceed_universe(self):
        evaluator = CrossSectionalEvaluator()

        with self.assertRaises(ValueError):
            evaluator.evaluate(
                predicted_returns=[0.10, 0.05],
                realized_returns=[0.08, 0.03],
                benchmark_returns=[0.01, 0.01],
                top_k=3,
            )

    def test_length_mismatch(self):
        evaluator = CrossSectionalEvaluator()

        with self.assertRaises(ValueError):
            evaluator.evaluate(
                predicted_returns=[0.10, 0.05],
                realized_returns=[0.08],
                benchmark_returns=[0.01, 0.01],
                top_k=1,
            )

    def test_universe_metrics(self):
        evaluator = CrossSectionalEvaluator()

        metrics = evaluator.evaluate(
            predicted_returns=[0.10, 0.05],
            realized_returns=[0.08, 0.02],
            benchmark_returns=[0.03, 0.01],
            top_k=1,
        )

        self.assertAlmostEqual(
            metrics.universe_mean_return,
            0.05,
        )

        self.assertAlmostEqual(
            metrics.universe_mean_excess_return,
            0.03,
        )


if __name__ == "__main__":
    unittest.main()
