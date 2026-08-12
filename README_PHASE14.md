# Phase 14 — Strategy Benchmark & Robustness

Phase 14 compares the frozen Phase 13 OOS model strategy against simple,
deterministic baselines using the exact same realized event-return
observations.

## Strategies

- `MODEL`: uses the already-created Phase 13 LONG / SHORT / NO_POSITION signal.
- `ALWAYS_LONG`: long every event with a valid realized return.
- `ALWAYS_SHORT`: short every event with a valid realized return.
- `NO_POSITION`: always stays out of the market.

## Rules

Phase 14 does not:

- retrain the model
- tune the probability threshold
- select a model using OOS results
- resplit the dataset
- alter Phase 13 signals
- use future outcomes to create signals

The comparison is therefore a robustness/benchmark layer over the frozen OOS
strategy.

## Metrics

For each strategy:

- total return
- average return
- win rate
- volatility
- Sharpe ratio
- trade count

The report also exposes model total-return deltas versus always-long and
always-short baselines.

This phase intentionally does not claim statistical significance. Formal
bootstrap/event-level inference is reserved for a later phase.
