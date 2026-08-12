# Phase 20 — Transaction Cost, Slippage & Execution Modeling

Phase 20 converts frozen gross portfolio returns into net returns under
explicit execution assumptions.

## Goals

- model commission/brokerage cost
- model slippage
- model spread friction
- model execution-delay cost
- compare gross versus net performance
- measure cost drag
- report gross and net drawdown
- report gross and net Sharpe-like metrics
- provide a descriptive break-even proportional cost rate

## Leakage and evaluation contract

Execution assumptions are supplied explicitly. This phase does not infer
cost parameters from future returns, optimize the strategy, retrain models,
or change portfolio weights.

The break-even cost rate is a descriptive sensitivity threshold based on
observed gross return and turnover. It is not used to select a strategy.

## Important limitation

This is a research execution model, not a broker simulator. It does not
model order-book depth, queue position, partial fills, price impact curves,
exchange fees by venue, or actual order routing.

Those capabilities belong to a later live/paper execution layer.
