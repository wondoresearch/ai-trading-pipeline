import unittest

from app.regime_robustness import (
    MarketRegimeClassifier,
    Regime,
    RegimeAnalysisStatus,
    RegimeObservation,
    RegimeRobustnessAnalyzer,
)


class TestRegimeRobustness(unittest.TestCase):
    def observations(self):
        values = [0.01, 0.01, 0.01, 0.01, 0.01, -0.01, -0.01, -0.01, -0.01, -0.01, 0.0, 0.0]
        return tuple(
            RegimeObservation(float(i), benchmark_return=v, strategy_return=v / 2)
            for i, v in enumerate(values)
        )

    def test_classifier_uses_only_trailing_history(self):
        obs = self.observations()
        labels = MarketRegimeClassifier(lookback=3, bull_threshold=0.005, bear_threshold=-0.005).classify(obs)
        self.assertEqual(labels[:3], (Regime.SIDEWAYS,) * 3)
        self.assertEqual(labels[3], Regime.BULL)
        self.assertEqual(labels[8], Regime.BEAR)

    def test_future_change_does_not_change_prior_regime(self):
        obs = self.observations()
        classifier = MarketRegimeClassifier(lookback=3, bull_threshold=0.005, bear_threshold=-0.005)
        original = classifier.classify(obs)
        altered = list(obs)
        altered[-1] = RegimeObservation(11.0, 0.5, 0.0)
        changed = classifier.classify(altered)
        self.assertEqual(original[:-1], changed[:-1])

    def test_custom_labels_are_isolated(self):
        obs = self.observations()[:6]
        labels = [Regime.BULL, Regime.BULL, Regime.BEAR, Regime.BEAR, Regime.SIDEWAYS, Regime.SIDEWAYS]
        report = RegimeRobustnessAnalyzer().analyze(obs, labels)
        self.assertEqual(report.status, RegimeAnalysisStatus.READY)
        self.assertEqual(report.regime_counts["BULL"], 2)
        self.assertEqual(report.regime_counts["BEAR"], 2)
        self.assertEqual(report.regime_counts["SIDEWAYS"], 2)

    def test_total_return_is_compounded(self):
        obs = (
            RegimeObservation(0, 0.0, 0.10),
            RegimeObservation(1, 0.0, -0.05),
        )
        labels = [Regime.BULL, Regime.BULL]
        report = RegimeRobustnessAnalyzer().analyze(obs, labels)
        bull = next(x for x in report.metrics if x.regime is Regime.BULL)
        self.assertAlmostEqual(bull.total_return, 1.10 * 0.95 - 1.0)

    def test_hit_rate(self):
        obs = tuple(
            RegimeObservation(float(i), 0.0, r)
            for i, r in enumerate([0.1, -0.1, 0.2, 0.0])
        )
        report = RegimeRobustnessAnalyzer().analyze(obs, [Regime.BULL] * 4)
        metric = report.metrics[0]
        self.assertAlmostEqual(metric.hit_rate, 0.5)

    def test_maximum_drawdown(self):
        obs = tuple(
            RegimeObservation(float(i), 0.0, r)
            for i, r in enumerate([0.10, -0.20, 0.05])
        )
        report = RegimeRobustnessAnalyzer().analyze(obs, [Regime.BULL] * 3)
        metric = report.metrics[0]
        self.assertLess(metric.maximum_drawdown, 0)
        self.assertAlmostEqual(metric.maximum_drawdown, -0.20, places=8)

    def test_single_observation_has_no_sharpe(self):
        obs = (RegimeObservation(0, 0.0, 0.01),)
        report = RegimeRobustnessAnalyzer().analyze(obs, [Regime.BULL])
        self.assertIsNone(report.metrics[0].sharpe_like)

    def test_unsorted_observations_are_invalid(self):
        obs = (
            RegimeObservation(2, 0.0, 0.01),
            RegimeObservation(1, 0.0, 0.01),
        )
        report = RegimeRobustnessAnalyzer().analyze(obs, [Regime.BULL, Regime.BULL])
        self.assertEqual(report.status, RegimeAnalysisStatus.INVALID_INPUT)

    def test_mismatched_labels_are_invalid(self):
        report = RegimeRobustnessAnalyzer().analyze(
            self.observations(), [Regime.BULL]
        )
        self.assertEqual(report.status, RegimeAnalysisStatus.INVALID_INPUT)

    def test_deterministic_json(self):
        obs = self.observations()
        analyzer = RegimeRobustnessAnalyzer()
        labels = MarketRegimeClassifier().classify(obs)
        a = analyzer.analyze(obs, labels)
        b = analyzer.analyze(obs, labels)
        self.assertEqual(a.to_json(), b.to_json())

    def test_no_observations_is_insufficient(self):
        report = RegimeRobustnessAnalyzer().analyze([])
        self.assertEqual(report.status, RegimeAnalysisStatus.INVALID_INPUT)


if __name__ == "__main__":
    unittest.main()
