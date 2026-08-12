import unittest

from app.portfolio_evaluation import (
    AssetObservation,
    PortfolioConfig,
    PortfolioEvaluator,
    PortfolioStatus,
)


class TestPortfolioEvaluator(unittest.TestCase):
    def observations(self):
        return (
            AssetObservation(0, "BBCA", 0.10, 1),
            AssetObservation(0, "BBRI", 0.05, 1),
            AssetObservation(1, "BBCA", -0.02, 1),
            AssetObservation(1, "BBRI", 0.04, -1),
            AssetObservation(2, "BBCA", 0.03, 0),
            AssetObservation(2, "BBRI", 0.02, 1),
        )

    def config(self):
        return PortfolioConfig(
            weights={"BBCA": 0.5, "BBRI": 0.5},
            max_gross_exposure=1.0,
            max_net_exposure=1.0,
            max_ticker_weight=0.5,
        )

    def test_portfolio_return_uses_explicit_weights_and_signals(self):
        report = PortfolioEvaluator().evaluate(self.observations(), self.config())
        self.assertEqual(report.status, PortfolioStatus.READY)
        # t0 = .5*.10 + .5*.05 = .075
        # t1 = .5*(-.02) + (-.5)*.04 = -.03
        # t2 = .5*.02 = .01
        expected = 1.075 * 0.97 * 1.01 - 1
        self.assertAlmostEqual(report.metrics.total_return, expected)

    def test_missing_asset_is_zero_exposure(self):
        obs = (
            AssetObservation(0, "BBCA", 0.10, 1),
            AssetObservation(1, "BBCA", 0.05, 1),
        )
        report = PortfolioEvaluator().evaluate(obs, self.config())
        self.assertEqual(report.status, PortfolioStatus.READY)

    def test_gross_exposure_limit_is_enforced(self):
        config = PortfolioConfig(
            weights={"BBCA": 0.7, "BBRI": 0.7},
            max_gross_exposure=1.0,
            max_net_exposure=2.0,
            max_ticker_weight=1.0,
        )
        report = PortfolioEvaluator().evaluate(self.observations(), config)
        self.assertEqual(report.status, PortfolioStatus.INVALID_INPUT)

    def test_net_exposure_limit_is_enforced(self):
        config = PortfolioConfig(
            weights={"BBCA": 0.6, "BBRI": 0.6},
            max_gross_exposure=2.0,
            max_net_exposure=1.0,
            max_ticker_weight=1.0,
        )
        report = PortfolioEvaluator().evaluate(self.observations(), config)
        self.assertEqual(report.status, PortfolioStatus.INVALID_INPUT)

    def test_ticker_concentration_limit_is_enforced(self):
        config = PortfolioConfig(
            weights={"BBCA": 0.8, "BBRI": 0.2},
            max_gross_exposure=1.0,
            max_net_exposure=1.0,
            max_ticker_weight=0.5,
        )
        report = PortfolioEvaluator().evaluate(self.observations(), config)
        self.assertEqual(report.status, PortfolioStatus.INVALID_INPUT)

    def test_attribution_sums_to_total_period_contribution(self):
        report = PortfolioEvaluator().evaluate(self.observations(), self.config())
        contribution = sum(x.contribution for x in report.attribution)
        # Attribution is additive in simple-period contributions, not compounded.
        expected = 0.075 - 0.03 + 0.01
        self.assertAlmostEqual(contribution, expected)

    def test_turnover_is_computed_from_signed_weight_changes(self):
        report = PortfolioEvaluator().evaluate(self.observations(), self.config())
        # t0 = 1.0; t1 changes BBRI from +.5 to -.5 => +1.0; t2
        # changes BBCA .5->0 and BBRI -.5->+.5 => 1.5; total = 3.5.
        self.assertAlmostEqual(report.metrics.turnover, 3.5)

    def test_risk_metrics_are_available(self):
        report = PortfolioEvaluator().evaluate(self.observations(), self.config())
        metrics = report.metrics
        self.assertGreaterEqual(metrics.volatility, 0)
        self.assertLessEqual(metrics.maximum_drawdown, 0)
        self.assertEqual(metrics.sample_size, 3)

    def test_chronological_order_is_required(self):
        obs = list(self.observations())
        obs[0], obs[-1] = obs[-1], obs[0]
        report = PortfolioEvaluator().evaluate(obs, self.config())
        self.assertEqual(report.status, PortfolioStatus.INVALID_INPUT)

    def test_unknown_ticker_is_explicit(self):
        obs = self.observations() + (AssetObservation(3, "BMRI", 0.01, 1),)
        report = PortfolioEvaluator().evaluate(obs, self.config())
        self.assertEqual(report.status, PortfolioStatus.INVALID_INPUT)

    def test_deterministic_json(self):
        evaluator = PortfolioEvaluator()
        a = evaluator.evaluate(self.observations(), self.config())
        b = evaluator.evaluate(self.observations(), self.config())
        self.assertEqual(a.to_json(), b.to_json())

    def test_empty_input_is_invalid(self):
        report = PortfolioEvaluator().evaluate([], self.config())
        self.assertEqual(report.status, PortfolioStatus.INVALID_INPUT)


if __name__ == "__main__":
    unittest.main()
