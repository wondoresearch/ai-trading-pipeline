import unittest
from sklearn.linear_model import LogisticRegression

from app.oos_evaluator import OOSEvaluator, OOSStatus
from app.trading_signal import TradingSignalEngine, Signal
from app.backtest_engine import BacktestEngine, BacktestStatus, TradeResult


class TestPhase13(unittest.TestCase):
    def data(self):
        x_train = [[-2], [-1], [1], [2], [-3], [3]]
        y_train = ["NEGATIVE", "NEGATIVE", "POSITIVE", "POSITIVE", "NEGATIVE", "POSITIVE"]
        model = LogisticRegression(random_state=42).fit(x_train, y_train)
        return model

    def test_oos_prediction(self):
        result = OOSEvaluator().evaluate(
            model=self.data(), x_oos=[[-2], [2]], y_oos=["NEGATIVE", "POSITIVE"],
            event_ids=["e1", "e2"], tickers=["BBCA", "BBRI"], threshold=0.5)
        self.assertEqual(result.status, OOSStatus.READY)
        self.assertEqual(result.sample_size, 2)

    def test_oos_does_not_fit_model(self):
        model = self.data()
        before = model.coef_.copy()
        OOSEvaluator().evaluate(model=model, x_oos=[[-2], [2]], y_oos=["NEGATIVE", "POSITIVE"],
                                event_ids=["e1", "e2"], tickers=["BBCA", "BBRI"])
        self.assertEqual(model.coef_.tolist(), before.tolist())

    def test_oos_invalid_lengths(self):
        result = OOSEvaluator().evaluate(
            model=self.data(), x_oos=[[-1]], y_oos=["NEGATIVE", "POSITIVE"],
            event_ids=["e1", "e2"], tickers=["BBCA", "BBRI"])
        self.assertEqual(result.status, OOSStatus.INVALID_INPUT)

    def test_signal_boundaries(self):
        engine = TradingSignalEngine()
        self.assertEqual(engine.generate("e", "BBCA", 0.8, 0.7).signal, Signal.LONG)
        self.assertEqual(engine.generate("e", "BBCA", 0.2, 0.7).signal, Signal.SHORT)
        self.assertEqual(engine.generate("e", "BBCA", 0.5, 0.7).signal, Signal.NO_POSITION)

    def test_missing_probability_is_no_position(self):
        result = TradingSignalEngine().generate("e", "BBCA", None, 0.7)
        self.assertEqual(result.signal, Signal.NO_POSITION)

    def test_backtest_costs(self):
        trades = (
            TradeResult("e1", "BBCA", "LONG", 0.05, 0.01, 0.01, 0.03),
            TradeResult("e2", "BBRI", "SHORT", -0.02, 0.01, 0.005, -0.035),
        )
        result = BacktestEngine().run(trades)
        self.assertEqual(result.status, BacktestStatus.READY)
        self.assertAlmostEqual(result.total_return, (1.03 * 0.965) - 1)

    def test_no_position_not_counted(self):
        trades = (
            TradeResult("e1", "BBCA", "NO_POSITION", None, 0, 0, None),
            TradeResult("e2", "BBRI", "LONG", 0.02, 0, 0, 0.02),
        )
        result = BacktestEngine().run(trades)
        self.assertEqual(result.number_of_trades, 1)

    def test_deterministic_json(self):
        trades = (TradeResult("e1", "BBCA", "LONG", 0.02, 0.001, 0.001, 0.018),)
        a = BacktestEngine().run(trades).to_json()
        b = BacktestEngine().run(trades).to_json()
        self.assertEqual(a, b)

    def test_oos_result_does_not_change_model_selection(self):
        # Phase 13 has no model-selection API; it consumes a frozen model only.
        model = self.data()
        before = model.coef_.tolist()
        OOSEvaluator().evaluate(model=model, x_oos=[[-100], [100]],
                                y_oos=["POSITIVE", "NEGATIVE"],
                                event_ids=["e1", "e2"], tickers=["BBCA", "BBRI"])
        self.assertEqual(model.coef_.tolist(), before)


if __name__ == "__main__":
    unittest.main()
