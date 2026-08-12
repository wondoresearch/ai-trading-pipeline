# Phase 17 — Market Regime / Condition Robustness

Phase 17 evaluates whether OOS strategy performance is concentrated in a
specific market condition.

## Contract

- regime classification uses only current/past benchmark observations
- no future benchmark return is used to classify an earlier observation
- no model retraining
- no threshold tuning
- no OOS-driven model selection
- chronological observations are required
- each regime is analyzed independently

The default classifier uses the mean benchmark return over a trailing lookback
window. Observations without sufficient history are classified as SIDEWAYS.

## Metrics per regime

- sample size
- compounded total return
- mean return
- volatility
- hit rate
- maximum drawdown
- Sharpe-like statistic when volatility and sample size permit

## Interpretation

The analysis is diagnostic rather than a claim of profitability. A strategy
that performs well overall but fails systematically in one major regime should
not be considered fully robust.

Phase 17 does not establish statistical significance under dependence, live
execution viability, transaction-cost robustness, or regime-detection accuracy.
Those remain separate research concerns.
