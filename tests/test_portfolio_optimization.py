import unittest

from app.portfolio_optimization import (
    OptimizationConfig,
    OptimizationStatus,
    PortfolioOptimizer,
)


class TestPortfolioOptimizer(unittest.TestCase):
    def inputs(self):
        tickers = ("BBCA", "BBRI", "BMRI")
        mu = (0.10, 0.08, 0.06)
        cov = (
            (0.04, 0.01, 0.008),
            (0.01, 0.03, 0.009),
            (0.008, 0.009, 0.02),
        )
        return tickers, mu, cov

    def test_optimization_is_ready(self):
        t, mu, cov = self.inputs()
        result = PortfolioOptimizer().optimize(t, mu, cov, OptimizationConfig())
        self.assertEqual(result.status, OptimizationStatus.READY)
        self.assertAlmostEqual(sum(result.weights), 1.0, places=8)

    def test_max_weight_is_respected(self):
        t, mu, cov = self.inputs()
        result = PortfolioOptimizer().optimize(
            t, mu, cov, OptimizationConfig(max_weight=0.5)
        )
        self.assertTrue(all(w <= 0.5 + 1e-10 for w in result.weights))

    def test_weights_are_long_only(self):
        t, mu, cov = self.inputs()
        result = PortfolioOptimizer().optimize(t, mu, cov, OptimizationConfig())
        self.assertTrue(all(w >= -1e-12 for w in result.weights))

    def test_higher_expected_return_receives_no_less_than_equal_case(self):
        t, mu, cov = self.inputs()
        result = PortfolioOptimizer().optimize(
            t, mu, cov,
            OptimizationConfig(risk_aversion=0.1, max_weight=0.6)
        )
        self.assertGreaterEqual(result.weights[0], result.weights[2])

    def test_risk_aversion_changes_solution(self):
        t, mu, cov = self.inputs()
        low = PortfolioOptimizer().optimize(
            t, mu, cov, OptimizationConfig(risk_aversion=0.1)
        )
        high = PortfolioOptimizer().optimize(
            t, mu, cov, OptimizationConfig(risk_aversion=10.0)
        )
        self.assertNotEqual(low.weights, high.weights)

    def test_objective_is_consistent(self):
        t, mu, cov = self.inputs()
        config = OptimizationConfig(risk_aversion=2.0)
        result = PortfolioOptimizer().optimize(t, mu, cov, config)
        expected = result.expected_return - config.risk_aversion * result.portfolio_variance
        self.assertAlmostEqual(result.objective, expected)

    def test_oos_returns_are_not_part_of_api(self):
        self.assertNotIn("oos_return", PortfolioOptimizer.optimize.__annotations__)

    def test_deterministic_result(self):
        t, mu, cov = self.inputs()
        config = OptimizationConfig()
        a = PortfolioOptimizer().optimize(t, mu, cov, config)
        b = PortfolioOptimizer().optimize(t, mu, cov, config)
        self.assertEqual(a.to_json(), b.to_json())

    def test_invalid_dimension_is_explicit(self):
        t, mu, cov = self.inputs()
        result = PortfolioOptimizer().optimize(t, mu[:-1], cov, OptimizationConfig())
        self.assertEqual(result.status, OptimizationStatus.INVALID_INPUT)

    def test_invalid_covariance_is_explicit(self):
        t, mu, cov = self.inputs()
        bad = list(map(list, cov))
        bad[0][1] = 0.99
        result = PortfolioOptimizer().optimize(t, mu, bad, OptimizationConfig())
        self.assertEqual(result.status, OptimizationStatus.INVALID_INPUT)

    def test_infeasible_weight_cap_is_explicit(self):
        t, mu, cov = self.inputs()
        result = PortfolioOptimizer().optimize(
            t, mu, cov, OptimizationConfig(max_weight=0.2)
        )
        self.assertEqual(result.status, OptimizationStatus.INVALID_INPUT)

    def test_invalid_config_is_explicit(self):
        t, mu, cov = self.inputs()
        result = PortfolioOptimizer().optimize(
            t, mu, cov, OptimizationConfig(risk_aversion=-1)
        )
        self.assertEqual(result.status, OptimizationStatus.INVALID_INPUT)

    def test_metadata_is_json_safe(self):
        t, mu, cov = self.inputs()
        result = PortfolioOptimizer().optimize(t, mu, cov, OptimizationConfig())
        self.assertIsInstance(result.to_json(), str)


if __name__ == "__main__":
    unittest.main()
