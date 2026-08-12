import unittest

from app.execution_model import (
    ExecutionConfig,
    ExecutionModel,
    ExecutionObservation,
    ExecutionStatus,
)


class TestExecutionModel(unittest.TestCase):
    def observations(self):
        return (
            ExecutionObservation(0, 0.05, 1.0, 1.0),
            ExecutionObservation(1, -0.02, 0.5, 1.0),
            ExecutionObservation(2, 0.03, 2.0, 0.5),
        )

    def test_zero_cost_preserves_gross_return(self):
        report = ExecutionModel().evaluate(
            self.observations(), ExecutionConfig()
        )
        self.assertEqual(report.status, ExecutionStatus.READY)
        self.assertAlmostEqual(
            report.metrics.net_total_return,
            report.metrics.gross_total_return,
        )
        self.assertAlmostEqual(report.metrics.total_execution_cost, 0)

    def test_commission_reduces_net_return(self):
        report = ExecutionModel().evaluate(
            self.observations(), ExecutionConfig(commission_rate=0.01)
        )
        self.assertLess(
            report.metrics.net_total_return,
            report.metrics.gross_total_return,
        )
        self.assertGreater(report.metrics.total_execution_cost, 0)

    def test_slippage_and_spread_are_included(self):
        report = ExecutionModel().evaluate(
            self.observations(),
            ExecutionConfig(slippage_rate=0.005, spread_rate=0.002),
        )
        expected = (1.0 + 0.5 + 2.0) * 0.007
        self.assertAlmostEqual(report.metrics.total_execution_cost, expected)

    def test_execution_delay_cost_uses_active_exposure(self):
        report = ExecutionModel().evaluate(
            self.observations(),
            ExecutionConfig(execution_delay_cost=0.01),
        )
        expected = (1.0 + 1.0 + 0.5) * 0.01
        self.assertAlmostEqual(report.metrics.total_execution_cost, expected)

    def test_cost_drag_is_net_minus_gross(self):
        report = ExecutionModel().evaluate(
            self.observations(), ExecutionConfig(commission_rate=0.01)
        )
        self.assertAlmostEqual(
            report.metrics.cost_drag,
            report.metrics.net_total_return
            - report.metrics.gross_total_return,
        )

    def test_net_drawdown_is_computed(self):
        report = ExecutionModel().evaluate(
            self.observations(), ExecutionConfig(commission_rate=0.01)
        )
        self.assertLessEqual(report.metrics.net_maximum_drawdown, 0)

    def test_sharpe_metrics_are_available(self):
        report = ExecutionModel().evaluate(
            self.observations(), ExecutionConfig()
        )
        self.assertIsNotNone(report.metrics.gross_sharpe_like)
        self.assertIsNotNone(report.metrics.net_sharpe_like)

    def test_break_even_cost_is_descriptive(self):
        report = ExecutionModel().evaluate(
            self.observations(), ExecutionConfig()
        )
        self.assertIsNotNone(report.metrics.break_even_cost_rate)
        self.assertGreater(report.metrics.break_even_cost_rate, 0)

    def test_unsorted_observations_are_invalid(self):
        obs = list(self.observations())
        obs.reverse()
        report = ExecutionModel().evaluate(obs, ExecutionConfig())
        self.assertEqual(report.status, ExecutionStatus.INVALID_INPUT)

    def test_duplicate_timestamps_are_invalid(self):
        obs = list(self.observations())
        obs[1] = ExecutionObservation(0, -0.02, 0.5, 1.0)
        report = ExecutionModel().evaluate(obs, ExecutionConfig())
        self.assertEqual(report.status, ExecutionStatus.INVALID_INPUT)

    def test_negative_execution_rate_is_invalid(self):
        report = ExecutionModel().evaluate(
            self.observations(), ExecutionConfig(commission_rate=-0.01)
        )
        self.assertEqual(report.status, ExecutionStatus.INVALID_INPUT)

    def test_empty_observations_are_invalid(self):
        report = ExecutionModel().evaluate([], ExecutionConfig())
        self.assertEqual(report.status, ExecutionStatus.INVALID_INPUT)

    def test_deterministic_json(self):
        model = ExecutionModel()
        config = ExecutionConfig(
            commission_rate=0.001,
            slippage_rate=0.002,
            spread_rate=0.001,
            execution_delay_cost=0.0005,
        )
        a = model.evaluate(self.observations(), config)
        b = model.evaluate(self.observations(), config)
        self.assertEqual(a.to_json(), b.to_json())


if __name__ == "__main__":
    unittest.main()
