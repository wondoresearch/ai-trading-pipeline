# Phase 19 — Portfolio-Level Evaluation & Risk Aggregation

Phase 19 moves the research pipeline from single-asset strategy evaluation to
portfolio-level evaluation.

## Goals

- combine frozen per-ticker OOS signals and returns
- apply explicit, predeclared portfolio weights
- enforce gross exposure, net exposure, and concentration limits
- calculate portfolio return and risk metrics
- calculate simple additive contribution by ticker
- calculate turnover from changes in signed target weights

## Leakage contract

Portfolio weights are supplied explicitly by the caller. This phase does not
optimize weights from OOS returns, does not retrain models, and does not use
future returns to determine positions.

Signals are evaluated at each timestamp. If an asset has no observation at a
timestamp, its exposure is treated as zero for that timestamp.

## Metrics

- compounded portfolio total return
- mean portfolio return
- volatility
- maximum drawdown
- Sharpe-like statistic
- average gross exposure
- average net exposure
- turnover
- additive ticker contribution

The contribution metric is intentionally additive in period P&L
contributions; it is not a compounded attribution decomposition.

## Interpretation

Portfolio-level robustness is a separate question from single-ticker
robustness. A set of individually attractive signals can still produce poor
portfolio behavior because of concentration, correlated exposures, turnover,
or conflicting positions.

This phase does not implement portfolio optimization, covariance-aware
position sizing, transaction-cost modeling beyond the supplied returns, or live
order execution.
