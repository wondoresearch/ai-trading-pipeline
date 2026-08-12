# Phase 16 — Walk-Forward / Temporal Robustness

Phase 16 evaluates the strategy through chronological train → validation →
OOS folds. Each fold trains only on observations before its validation and OOS
windows, then evaluates the immediately following OOS window.

## Contract

- chronological data only
- no random shuffle
- no random K-fold
- training never receives OOS rows
- validation always precedes OOS
- each fold receives its own frozen model artifact and threshold
- realized OOS returns are used only after prediction
- fold results remain individually inspectable

## Outputs

Per fold:

- train/validation/OOS boundaries
- sample sizes
- frozen threshold
- OOS returns
- compounded OOS return
- mean OOS return
- positive-return ratio

Aggregate:

- compounded OOS return
- mean OOS return
- positive-fold ratio
- best fold
- worst fold

A positive-fold ratio of 0.8, for example, means 80% of completed folds had
positive compounded OOS returns.

## Important limitation

This phase validates temporal stability; it does not by itself establish
statistical significance, live profitability, or immunity to regime change.
Model/threshold selection must remain inside each training/validation fold.
