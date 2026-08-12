import unittest

from app.ml_benchmark import MLBenchmark, BenchmarkStatus


class TestMLBenchmark(unittest.TestCase):
    feature_names = ("x1", "x2")

    def data(self):
        x_train = [
            [-3, -2], [-2, -1], [-1, -2],
            [1, 1], [2, 1], [3, 2],
            [-2, 0], [2, 0],
        ]
        y_train = ["NEGATIVE", "NEGATIVE", "NEGATIVE", "POSITIVE",
                   "POSITIVE", "POSITIVE", "NEGATIVE", "POSITIVE"]
        x_val = [[-2, -1], [-1, -1], [1, 1], [2, 2]]
        y_val = ["NEGATIVE", "NEGATIVE", "POSITIVE", "POSITIVE"]
        x_test = [[-3, -1], [3, 1], [-1, 0], [1, 0]]
        y_test = ["NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE"]
        return x_train, y_train, x_val, y_val, x_test, y_test

    def test_all_baselines_run(self):
        data = self.data()
        report = MLBenchmark().run(
            x_train=data[0], y_train=data[1],
            x_validation=data[2], y_validation=data[3],
            x_test=data[4], y_test=data[5],
            feature_names=self.feature_names,
        )
        self.assertEqual(report.status, BenchmarkStatus.READY)
        self.assertEqual(len(report.models), 4)
        self.assertIn(report.selected_model, {
            "dummy", "logistic_regression", "random_forest", "gradient_boosting"
        })

    def test_selection_uses_validation_not_test(self):
        data = self.data()
        report = MLBenchmark().run(
            x_train=data[0], y_train=data[1],
            x_validation=data[2], y_validation=data[3],
            x_test=data[4], y_test=["POSITIVE"] * 4,
            feature_names=self.feature_names,
        )
        # Test labels are intentionally changed; selection must still be
        # determined from validation.
        self.assertIn(report.selected_model, {
            "dummy", "logistic_regression", "random_forest", "gradient_boosting"
        })
        selected = next(m for m in report.models if m.model_name == report.selected_model)
        best_validation = max(m.validation.f1 for m in report.models)
        self.assertEqual(selected.validation.f1, best_validation)

    def test_deterministic_repeated_run(self):
        data = self.data()
        benchmark = MLBenchmark()
        a = benchmark.run(
            x_train=data[0], y_train=data[1],
            x_validation=data[2], y_validation=data[3],
            x_test=data[4], y_test=data[5],
            feature_names=self.feature_names,
        )
        b = benchmark.run(
            x_train=data[0], y_train=data[1],
            x_validation=data[2], y_validation=data[3],
            x_test=data[4], y_test=data[5],
            feature_names=self.feature_names,
        )
        self.assertEqual(a.to_json(), b.to_json())

    def test_invalid_dimensions_are_explicit(self):
        data = self.data()
        report = MLBenchmark().run(
            x_train=data[0], y_train=data[1],
            x_validation=data[2], y_validation=data[3],
            x_test=[[1]], y_test=data[5],
            feature_names=self.feature_names,
        )
        self.assertEqual(report.status, BenchmarkStatus.INVALID_INPUT)

    def test_missing_split_is_explicit(self):
        data = self.data()
        report = MLBenchmark().run(
            x_train=data[0], y_train=data[1],
            x_validation=[], y_validation=[],
            x_test=data[4], y_test=data[5],
            feature_names=self.feature_names,
        )
        self.assertEqual(report.status, BenchmarkStatus.INVALID_INPUT)

    def test_metric_values_are_bounded(self):
        data = self.data()
        report = MLBenchmark().run(
            x_train=data[0], y_train=data[1],
            x_validation=data[2], y_validation=data[3],
            x_test=data[4], y_test=data[5],
            feature_names=self.feature_names,
        )
        for model in report.models:
            for metrics in (model.validation, model.test):
                for name in (
                    "accuracy", "balanced_accuracy", "precision", "recall",
                    "f1", "roc_auc", "pr_auc"
                ):
                    value = getattr(metrics, name)
                    if value is not None:
                        self.assertGreaterEqual(value, 0.0)
                        self.assertLessEqual(value, 1.0)

    def test_custom_model_subset(self):
        data = self.data()
        report = MLBenchmark().run(
            x_train=data[0], y_train=data[1],
            x_validation=data[2], y_validation=data[3],
            x_test=data[4], y_test=data[5],
            feature_names=self.feature_names,
            models=("dummy", "logistic_regression"),
        )
        self.assertEqual([m.model_name for m in report.models],
                         ["dummy", "logistic_regression"])

    def test_invalid_model_name(self):
        data = self.data()
        report = MLBenchmark().run(
            x_train=data[0], y_train=data[1],
            x_validation=data[2], y_validation=data[3],
            x_test=data[4], y_test=data[5],
            feature_names=self.feature_names,
            models=("does_not_exist",),
        )
        self.assertEqual(report.status, BenchmarkStatus.INVALID_INPUT)


if __name__ == "__main__":
    unittest.main()
