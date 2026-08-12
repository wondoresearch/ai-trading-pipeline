import unittest

from app.stress_robustness import (
    OOSObservation,
    StressConfig,
    StressScenario,
    StressStatus,
    StressTester,
)


class TestStressTester(unittest.TestCase):
    def observations(self):
        return (
            OOSObservation(0, 0.10, 1),
            OOSObservation(1, 0.05, 1),
            OOSObservation(2, -0.02, -1),
            OOSObservation(3, 0.03, 1),
        )

    def test_baseline_is_present(self):
        report = StressTester().run(self.observations(), [StressConfig()])
        self.assertEqual(report.status, StressStatus.READY)
        self.assertIsNotNone(report.baseline)
        self.assertAlmostEqual(report.baseline.total_return, 1.10 * 1.05 * 0.98 * 1.03 - 1)

    def test_return_shock_reduces_positive_returns(self):
        report = StressTester().run(
            self.observations(), [StressConfig(return_shock=0.5)]
        )
        result = report.scenarios[0]
        self.assertEqual(result.scenario, StressScenario.RETURN_SHOCK)
        self.assertLess(result.total_return, report.baseline.total_return)

    def test_transaction_cost_reduces_return(self):
        report = StressTester().run(
            self.observations(), [StressConfig(transaction_cost=0.01)]
        )
        result = report.scenarios[0]
        self.assertEqual(result.scenario, StressScenario.TRANSACTION_COST)
        self.assertLess(result.total_return, report.baseline.total_return)

    def test_signal_noise_is_deterministic(self):
        config = StressConfig(signal_flip_rate=0.5)
        a = StressTester().run(self.observations(), [config])
        b = StressTester().run(self.observations(), [config])
        self.assertEqual(a.to_json(), b.to_json())

    def test_signal_noise_is_labeled(self):
        report = StressTester().run(
            self.observations(), [StressConfig(signal_flip_rate=0.5)]
        )
        self.assertEqual(report.scenarios[0].scenario, StressScenario.SIGNAL_NOISE)

    def test_baseline_is_not_used_to_change_observations(self):
        obs = self.observations()
        before = tuple(obs)
        StressTester().run(obs, [StressConfig(return_shock=0.2)])
        self.assertEqual(obs, before)

    def test_mixed_stress_is_allowed_and_preserved_in_metrics(self):
        report = StressTester().run(
            self.observations(),
            [StressConfig(return_shock=0.2, transaction_cost=0.01)],
        )
        self.assertEqual(report.status, StressStatus.READY)
        self.assertLess(report.scenarios[0].total_return, report.baseline.total_return)

    def test_invalid_order_is_explicit(self):
        obs = list(self.observations())
        obs[0], obs[1] = obs[1], obs[0]
        report = StressTester().run(obs, [StressConfig(return_shock=0.1)])
        self.assertEqual(report.status, StressStatus.INVALID_INPUT)

    def test_invalid_configuration_is_explicit(self):
        report = StressTester().run(
            self.observations(), [StressConfig(return_shock=1.0)]
        )
        self.assertEqual(report.status, StressStatus.INVALID_INPUT)

    def test_deterministic_json(self):
        configs = [
            StressConfig(return_shock=0.1),
            StressConfig(signal_flip_rate=0.25),
            StressConfig(transaction_cost=0.005),
        ]
        a = StressTester().run(self.observations(), configs)
        b = StressTester().run(self.observations(), configs)
        self.assertEqual(a.to_json(), b.to_json())

    def test_no_configs_is_insufficient(self):
        report = StressTester().run(self.observations(), [])
        self.assertEqual(report.status, StressStatus.INSUFFICIENT_DATA)


if __name__ == "__main__":
    unittest.main()
