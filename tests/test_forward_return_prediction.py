import unittest

from app.forward_return_prediction import (
    ForwardReturnPredictor,
    PredictionStatus,
)


class TestForwardReturnPredictor(unittest.TestCase):

    def data(self):
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

        return x_train, y_train, x_test, y_test, tickers

    def test_training_ready(self):
        data = self.data()

        report = ForwardReturnPredictor().train_and_evaluate(
            x_train=data[0],
            y_train=data[1],
            x_test=data[2],
            y_test=data[3],
            tickers=data[4],
            horizon="t5",
        )

        self.assertEqual(report.status, PredictionStatus.READY)
        self.assertEqual(report.horizon, "t5")
        self.assertEqual(report.model_name, "ridge")
        self.assertIsNotNone(report.metrics)
        self.assertEqual(len(report.predictions), 4)

    def test_tickers_are_preserved(self):
        data = self.data()

        report = ForwardReturnPredictor().train_and_evaluate(
            x_train=data[0],
            y_train=data[1],
            x_test=data[2],
            y_test=data[3],
            tickers=data[4],
            horizon="t5",
        )

        self.assertEqual(
            [item.ticker for item in report.predictions],
            data[4],
        )

    def test_directional_accuracy_is_bounded(self):
        data = self.data()

        report = ForwardReturnPredictor().train_and_evaluate(
            x_train=data[0],
            y_train=data[1],
            x_test=data[2],
            y_test=data[3],
            tickers=data[4],
            horizon="t5",
        )

        self.assertGreaterEqual(
            report.metrics.directional_accuracy,
            0.0,
        )
        self.assertLessEqual(
            report.metrics.directional_accuracy,
            1.0,
        )

    def test_supported_horizons(self):
        data = self.data()

        for horizon in ("t1", "t3", "t5", "t10"):
            report = ForwardReturnPredictor().train_and_evaluate(
                x_train=data[0],
                y_train=data[1],
                x_test=data[2],
                y_test=data[3],
                tickers=data[4],
                horizon=horizon,
            )

            self.assertEqual(
                report.status,
                PredictionStatus.READY,
            )

    def test_invalid_horizon(self):
        data = self.data()

        report = ForwardReturnPredictor().train_and_evaluate(
            x_train=data[0],
            y_train=data[1],
            x_test=data[2],
            y_test=data[3],
            tickers=data[4],
            horizon="t7",
        )

        self.assertEqual(
            report.status,
            PredictionStatus.INVALID_INPUT,
        )

    def test_length_mismatch_is_invalid(self):
        data = self.data()

        report = ForwardReturnPredictor().train_and_evaluate(
            x_train=data[0],
            y_train=data[1],
            x_test=data[2],
            y_test=data[3],
            tickers=data[4][:-1],
            horizon="t5",
        )

        self.assertEqual(
            report.status,
            PredictionStatus.INVALID_INPUT,
        )


if __name__ == "__main__":
    unittest.main()
