import unittest

from app.oos_robustness import (
    OOSRobustnessValidator,
    RobustnessStatus,
)


class TestOOSRobustness(unittest.TestCase):
    def returns(self):
        return (
            0.03, 0.02, -0.01, 0.04, 0.01,
            0.02, -0.005, 0.03, 0.015, 0.025,
            0.01, -0.008,
        )

    def test_ready_with_sufficient_data(self):
        report = OOSRobustnessValidator().validate(
            self.returns(), bootstrap_iterations=100, permutation_iterations=100
        )
        self.assertEqual(report.status, RobustnessStatus.READY)
        self.assertEqual(report.sample_size, 12)
        self.assertIsNotNone(report.bootstrap_mean_ci_low)
        self.assertIsNotNone(report.bootstrap_mean_ci_high)
        self.assertIsNotNone(report.permutation_p_value)

    def test_insufficient_data_is_explicit(self):
        report = OOSRobustnessValidator().validate(
            (0.01, 0.02), min_observations=10
        )
        self.assertEqual(report.status, RobustnessStatus.INSUFFICIENT_DATA)

    def test_invalid_nan_is_explicit(self):
        report = OOSRobustnessValidator().validate((0.01, float("nan")))
        self.assertEqual(report.status, RobustnessStatus.INVALID_INPUT)

    def test_invalid_iterations_are_explicit(self):
        report = OOSRobustnessValidator().validate(
            self.returns(), bootstrap_iterations=0
        )
        self.assertEqual(report.status, RobustnessStatus.INVALID_INPUT)

    def test_ci_is_ordered(self):
        report = OOSRobustnessValidator().validate(
            self.returns(), bootstrap_iterations=200, permutation_iterations=200
        )
        self.assertLessEqual(
            report.bootstrap_mean_ci_low,
            report.bootstrap_mean_ci_high,
        )

    def test_p_value_is_bounded(self):
        report = OOSRobustnessValidator().validate(
            self.returns(), bootstrap_iterations=100, permutation_iterations=100
        )
        self.assertGreaterEqual(report.permutation_p_value, 0.0)
        self.assertLessEqual(report.permutation_p_value, 1.0)

    def test_same_seed_is_deterministic(self):
        engine = OOSRobustnessValidator()
        a = engine.validate(
            self.returns(), bootstrap_iterations=200, permutation_iterations=200,
            seed=123
        )
        b = engine.validate(
            self.returns(), bootstrap_iterations=200, permutation_iterations=200,
            seed=123
        )
        self.assertEqual(a.to_json(), b.to_json())

    def test_different_seed_can_change_resampling(self):
        engine = OOSRobustnessValidator()
        a = engine.validate(
            self.returns(), bootstrap_iterations=200, permutation_iterations=200,
            seed=123
        )
        b = engine.validate(
            self.returns(), bootstrap_iterations=200, permutation_iterations=200,
            seed=456
        )
        self.assertNotEqual(a.to_json(), b.to_json())

    def test_total_return_is_compounded(self):
        report = OOSRobustnessValidator().validate(
            (0.10, 0.10) + (0.0,) * 8,
            bootstrap_iterations=50, permutation_iterations=50,
        )
        self.assertAlmostEqual(report.observed_total_return, 0.21)

    def test_fingerprint_is_deterministic(self):
        engine = OOSRobustnessValidator()
        self.assertEqual(
            engine.fingerprint(self.returns()),
            engine.fingerprint(self.returns()),
        )

    def test_validation_does_not_mutate_input(self):
        values = list(self.returns())
        before = list(values)
        OOSRobustnessValidator().validate(
            values, bootstrap_iterations=100, permutation_iterations=100
        )
        self.assertEqual(values, before)


if __name__ == "__main__":
    unittest.main()
