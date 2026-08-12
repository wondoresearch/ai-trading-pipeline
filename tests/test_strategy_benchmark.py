import unittest

from app.strategy_benchmark import (
    BenchmarkObservation,
    BenchmarkStatus,
    BenchmarkStrategy,
    StrategyBenchmarkEngine,
)


class TestStrategyBenchmark(unittest.TestCase):
    def observations(self):
        return (
            BenchmarkObservation("e1", "BBCA", "LONG", 0.03),
            BenchmarkObservation("e2", "BBRI", "SHORT", -0.02),
            BenchmarkObservation("e3", "BMRI", "NO_POSITION", 0.04),
            BenchmarkObservation("e4", "TLKM", "LONG", -0.01),
        )

    def test_all_baselines_run_on_same_observations(self):
        result = StrategyBenchmarkEngine().compare(self.observations())
        self.assertEqual(result.status, BenchmarkStatus.READY)
        self.assertEqual(result.observations, 4)
        self.assertEqual(
            {item.strategy for item in result.strategies},
            {
                BenchmarkStrategy.MODEL,
                BenchmarkStrategy.ALWAYS_LONG,
                BenchmarkStrategy.ALWAYS_SHORT,
                BenchmarkStrategy.NO_POSITION,
            },
        )

    def test_model_return_uses_only_model_signal(self):
        result = StrategyBenchmarkEngine().compare(self.observations())
        model = next(x for x in result.strategies if x.strategy == BenchmarkStrategy.MODEL)
        expected = (1.03 * 1.02 * 1.0 * 0.99) - 1.0
        self.assertAlmostEqual(model.total_return, expected)

    def test_always_long_and_short_are_opposites_per_event(self):
        result = StrategyBenchmarkEngine().compare(self.observations())
        long = next(x for x in result.strategies if x.strategy == BenchmarkStrategy.ALWAYS_LONG)
        short = next(x for x in result.strategies if x.strategy == BenchmarkStrategy.ALWAYS_SHORT)
        self.assertNotEqual(long.total_return, short.total_return)

    def test_no_position_has_zero_return(self):
        result = StrategyBenchmarkEngine().compare(
            self.observations(),
            strategies=(BenchmarkStrategy.NO_POSITION,),
        )
        metric = result.strategies[0]
        self.assertEqual(metric.total_return, 0.0)
        self.assertEqual(metric.trades, 0)

    def test_missing_realized_return_is_excluded(self):
        observations = self.observations() + (
            BenchmarkObservation("e5", "ASII", "LONG", None),
        )
        result = StrategyBenchmarkEngine().compare(observations)
        self.assertEqual(result.observations, 5)
        model = next(x for x in result.strategies if x.strategy == BenchmarkStrategy.MODEL)
        self.assertEqual(model.observations, 4)

    def test_duplicate_event_is_invalid(self):
        observations = self.observations() + (
            BenchmarkObservation("e1", "ASII", "LONG", 0.01),
        )
        result = StrategyBenchmarkEngine().compare(observations)
        self.assertEqual(result.status, BenchmarkStatus.INVALID_INPUT)

    def test_model_minus_baseline_is_reported(self):
        result = StrategyBenchmarkEngine().compare(self.observations())
        self.assertIsNotNone(result.model_minus_always_long)
        self.assertIsNotNone(result.model_minus_always_short)

    def test_deterministic_json(self):
        engine = StrategyBenchmarkEngine()
        self.assertEqual(
            engine.compare(self.observations()).to_json(),
            engine.compare(self.observations()).to_json(),
        )

    def test_benchmark_does_not_change_signal_inputs(self):
        observations = self.observations()
        before = tuple(o.model_signal for o in observations)
        StrategyBenchmarkEngine().compare(observations)
        self.assertEqual(tuple(o.model_signal for o in observations), before)

    def test_custom_strategy_subset(self):
        result = StrategyBenchmarkEngine().compare(
            self.observations(),
            strategies=(BenchmarkStrategy.MODEL, BenchmarkStrategy.NO_POSITION),
        )
        self.assertEqual(
            tuple(x.strategy for x in result.strategies),
            (BenchmarkStrategy.MODEL, BenchmarkStrategy.NO_POSITION),
        )


if __name__ == "__main__":
    unittest.main()
