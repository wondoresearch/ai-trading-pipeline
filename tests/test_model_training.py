import json
import tempfile
import unittest

from app.model_training import ModelTrainer, TrainingStatus


class TestModelTrainer(unittest.TestCase):
        def data(self):
            x_train = [[-3, -2], [-2, -1], [-1, -2], [1, 1], [2, 1], [3, 2], [-2, 0], [2, 0]]
            y_train = ["NEGATIVE", "NEGATIVE", "NEGATIVE", "POSITIVE",
                       "POSITIVE", "POSITIVE", "NEGATIVE", "POSITIVE"]
            x_val = [[-2, -1], [-1, -1], [1, 1], [2, 2]]
            y_val = ["NEGATIVE", "NEGATIVE", "POSITIVE", "POSITIVE"]
            x_test = [[-3, -1], [3, 1], [-1, 0], [1, 0]]
            y_test = ["NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE"]
            return x_train, y_train, x_val, y_val, x_test, y_test

        def make_report(self, family="logistic_regression", **kwargs):
            d = self.data()
            return ModelTrainer().train(
                model_family=family,
                x_train=d[0], y_train=d[1],
                x_validation=d[2], y_validation=d[3],
                x_test=d[4], y_test=d[5],
                feature_names=("x1", "x2"),
                **kwargs,
            )

        def test_training_ready(self):
            report = self.make_report()
            self.assertEqual(report.status, TrainingStatus.READY)
            self.assertIsNotNone(report.selected_candidate)
            self.assertIsNotNone(report.test)

        def test_selection_is_validation_based(self):
            normal = self.make_report()
            d = self.data()
            altered_test = ModelTrainer().train(
                model_family="logistic_regression",
                x_train=d[0], y_train=d[1],
                x_validation=d[2], y_validation=d[3],
                x_test=d[4], y_test=["POSITIVE"] * 4,
                feature_names=("x1", "x2"),
            )
            self.assertEqual(normal.selected_candidate, altered_test.selected_candidate)

        def test_custom_small_grid(self):
            report = self.make_report(
                parameter_grid=(
                    {"C": 0.1, "max_iter": 1000, "class_weight": None},
                    {"C": 10.0, "max_iter": 1000, "class_weight": None},
                )
            )
            self.assertEqual(len(report.candidates), 2)

        def test_threshold_selection_uses_validation(self):
            report = self.make_report(
                threshold_candidates=(0.3, 0.5, 0.7),
                selection_metric="f1",
            )
            self.assertIn(report.artifact_metadata["threshold"], (0.3, 0.5, 0.7))

        def test_deterministic_training(self):
            a = self.make_report()
            b = self.make_report()
            self.assertEqual(a.to_json(), b.to_json())

        def test_invalid_input(self):
            d = self.data()
            report = ModelTrainer().train(
                model_family="logistic_regression",
                x_train=d[0], y_train=d[1],
                x_validation=d[2], y_validation=[],
                x_test=d[4], y_test=d[5],
                feature_names=("x1", "x2"),
            )
            self.assertEqual(report.status, TrainingStatus.INVALID_INPUT)

        def test_unsupported_model_is_explicit(self):
            report = self.make_report(family="unknown")
            self.assertEqual(report.status, TrainingStatus.INVALID_INPUT)

        def test_metadata_is_json_safe(self):
            report = self.make_report()
            payload = report.to_json()
            json.loads(payload)
            self.assertNotIn("model", report.artifact_metadata)

        def test_metadata_can_be_saved(self):
            report = self.make_report()
            with tempfile.NamedTemporaryFile(suffix=".json") as handle:
                ModelTrainer.save_metadata(report, handle.name)
                handle.seek(0)
                data = json.load(handle)
                self.assertEqual(data["status"], "READY")


if __name__ == "__main__":
    unittest.main()
