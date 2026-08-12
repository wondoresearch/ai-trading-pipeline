# Phase 13 — Out-of-Sample Backtesting

Phase 13 evaluates a frozen Phase 12 model on an untouched out-of-sample
dataset. It performs no fitting, hyperparameter tuning, threshold tuning,
feature engineering, or data resplitting.

## Components

- `app/oos_evaluator.py`: blind prediction/evaluation.
- `app/trading_signal.py`: converts the frozen probability threshold into
  LONG / SHORT / NO_POSITION.
- `app/backtest_engine.py`: event-level gross/net return and performance
  metrics with explicit transaction cost and slippage.

## Economic metrics

- total return
- annualized return
- volatility
- Sharpe ratio
- maximum drawdown
- win rate
- profit factor
- number of trades
- turnover

## Boundary

The threshold is an input inherited from Phase 12. Phase 13 never searches
for a better threshold using OOS outcomes.

The backtest is event-driven. A trade is created from an OOS prediction and
its canonical event return; no future information is used to create the
signal.

Do not commit/tag/push until targeted and full regression tests pass.
