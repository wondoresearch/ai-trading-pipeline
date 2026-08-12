import unittest

from app.walk_forward import (
    FoldPlan,
    TemporalRow,
    TrainedFoldModel,
    WalkForwardEvaluator,
    WalkForwardStatus,
)


class TestWalkForwardEvaluator(unittest.TestCase):
    def rows(self, n=18):
        return tuple(
            TemporalRow(
                timestamp=float(i),
                features=(float(i),),
                label="POSITIVE" if i % 2 else "NEGATIVE",
                realized_return=0.01 if i % 3 else -0.005,
            )
            for i in range(n)
        )

    def trainer(self, calls):
        def train(train_rows, validation_rows):
            calls.append((
                train_rows[0].timestamp,
                train_rows[-1].timestamp,
                validation_rows[0].timestamp,
                validation_rows[-1].timestamp,
            ))
            return TrainedFoldModel(
                predict_signal=lambda features: "LONG",
                threshold=0.5,
            )
        return train

    def test_temporal_folds_are_ordered(self):
        calls = []
        report = WalkForwardEvaluator().evaluate(
            self.rows(),
            FoldPlan(train_size=6, validation_size=3, oos_size=3),
            self.trainer(calls),
        )
        self.assertEqual(report.status, WalkForwardStatus.READY)
        self.assertGreaterEqual(len(report.folds), 3)
        for fold in report.folds:
            self.assertLess(fold.train_end, fold.validation_start)
            self.assertLess(fold.validation_end, fold.oos_start)

    def test_validation_is_before_oos(self):
        calls = []
        report = WalkForwardEvaluator().evaluate(
            self.rows(),
            FoldPlan(6, 3, 3),
            self.trainer(calls),
        )
        for call, fold in zip(calls, report.folds):
            self.assertEqual(call[1], fold.train_end)
            self.assertEqual(call[3], fold.validation_end)
            self.assertLess(fold.validation_end, fold.oos_start)

    def test_threshold_is_frozen_into_fold(self):
        report = WalkForwardEvaluator().evaluate(
            self.rows(),
            FoldPlan(6, 3, 3),
            self.trainer([]),
        )
        self.assertTrue(all(f.threshold == 0.5 for f in report.folds))

    def test_no_future_oos_row_is_passed_to_trainer(self):
        seen = []

        def train(train_rows, validation_rows):
            seen.append(max(x.timestamp for x in validation_rows))
            return TrainedFoldModel(lambda features: "LONG", threshold=0.4)

        report = WalkForwardEvaluator().evaluate(
            self.rows(), FoldPlan(6, 3, 3), train
        )
        for maximum_validation_time, fold in zip(seen, report.folds):
            self.assertLess(maximum_validation_time, fold.oos_start)

    def test_positive_fold_ratio(self):
        report = WalkForwardEvaluator().evaluate(
            self.rows(), FoldPlan(6, 3, 3), self.trainer([])
        )
        expected = sum(f.oos_total_return > 0 for f in report.folds) / len(report.folds)
        self.assertAlmostEqual(report.positive_fold_ratio, expected)

    def test_aggregate_return_is_compounded(self):
        report = WalkForwardEvaluator().evaluate(
            self.rows(), FoldPlan(6, 3, 3), self.trainer([])
        )
        returns = [r for f in report.folds for r in f.oos_returns]
        expected = 1.0
        for value in returns:
            expected *= 1.0 + value
        self.assertAlmostEqual(report.aggregate_oos_total_return, expected - 1.0)

    def test_short_signal_flips_realized_return(self):
        def train(train_rows, validation_rows):
            return TrainedFoldModel(lambda features: "SHORT")

        report = WalkForwardEvaluator().evaluate(
            self.rows(), FoldPlan(6, 3, 3), train
        )
        self.assertTrue(all(r <= 0.01 for f in report.folds for r in f.oos_returns))

    def test_no_complete_fold_is_explicit(self):
        report = WalkForwardEvaluator().evaluate(
            self.rows(8), FoldPlan(6, 3, 3), self.trainer([])
        )
        self.assertEqual(report.status, WalkForwardStatus.INSUFFICIENT_FOLD_DATA)

    def test_unsorted_rows_are_invalid(self):
        rows = list(self.rows())
        rows[1], rows[2] = rows[2], rows[1]
        report = WalkForwardEvaluator().evaluate(
            rows, FoldPlan(6, 3, 3), self.trainer([])
        )
        self.assertEqual(report.status, WalkForwardStatus.INVALID_INPUT)

    def test_duplicate_timestamps_are_invalid(self):
        rows = list(self.rows())
        rows[5] = TemporalRow(4.0, rows[5].features, rows[5].label, rows[5].realized_return)
        report = WalkForwardEvaluator().evaluate(
            rows, FoldPlan(6, 3, 3), self.trainer([])
        )
        self.assertEqual(report.status, WalkForwardStatus.INVALID_INPUT)

    def test_deterministic_serialization(self):
        report = WalkForwardEvaluator().evaluate(
            self.rows(), FoldPlan(6, 3, 3), self.trainer([])
        )
        self.assertEqual(report.to_json(), report.to_json())


if __name__ == "__main__":
    unittest.main()
