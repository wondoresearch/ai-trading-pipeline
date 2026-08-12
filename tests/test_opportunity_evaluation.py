import unittest

from app.opportunity_evaluation import OpportunityEvaluator


class TestOpportunityEvaluator(unittest.TestCase):

    def test_integrated_evaluation(self):
        x_train = [
            [-3.0, -2.0],
            [-2.0, -1.0],
            [-1.0, -2.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [-2.0, 0.0],
            [2.0, 0.0],
        ]

        y_train = [
            -0.030,
            -0.020,
            -0.015,
             0.015,
             0.020,
             0.030,
            -0.010,
             0.010,
        ]

        x_test = [
            [-3.0, -1.0],
            [3.0, 1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ]

        y_test = [
            -0.025,
             0.025,
            -0.008,
             0.008,
        ]

        tickers = [
            "BBCA",
            "BBRI",
            "BMRI",
            "BBNI",
        ]

        benchmark_returns = [
            0.005,
            0.005,
            0.005,
            0.005,
        ]

        report = OpportunityEvaluator().evaluate(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            tickers=tickers,
            benchmark_returns=benchmark_returns,
            horizon="t5",
            top_k=2,
        )

        self.assertEqual(
            report.prediction.status.value,
            "READY",
        )

        self.assertEqual(
            len(report.prediction.predictions),
            4,
        )

        self.assertGreaterEqual(
            report.cross_sectional.hit_rate,
            0.0,
        )

        self.assertLessEqual(
            report.cross_sectional.hit_rate,
            1.0,
        )

        self.assertEqual(
            report.prediction.horizon,
            "t5",
        )


if __name__ == "__main__":
    unittest.main()
