# Phase 15 — OOS Statistical Robustness

Phase 15 validates the frozen Phase 13/14 OOS strategy statistically without
changing the model, threshold, signals, or train/validation/test boundaries.

## Methods

### 1. Bootstrap confidence interval

Resamples the observed OOS event returns with replacement and estimates a
confidence interval for the mean realized return.

### 2. Sign-flip permutation test

Tests the null hypothesis that the OOS mean return is centered at zero by
randomly flipping the sign of each observed event return.

The reported p-value uses a +1 correction:

`(extreme + 1) / (iterations + 1)`

### 3. Compounded OOS return

The observed total return is:

`product(1 + return_i) - 1`

## Leakage boundary

This phase consumes realized OOS returns only.

It does NOT:

- retrain a model
- tune a threshold
- select a strategy
- modify OOS signals
- inspect training/validation outcomes
- use future information to create a signal

## Interpretation

A p-value below 0.05 is reported as statistically significant under this
specific sign-flip null. It is not evidence that the strategy will remain
profitable live, and it does not correct for multiple strategy/model searches.

A later phase can add block bootstrap, dependence-aware inference, multiple
testing correction, and walk-forward evaluation.
