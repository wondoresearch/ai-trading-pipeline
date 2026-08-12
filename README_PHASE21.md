# Phase 21 — Constrained Portfolio Optimization

Phase 21 determines portfolio weights from explicitly supplied estimation
statistics.

## Objective

Maximize:

`expected_return - risk_aversion * portfolio_variance`

subject to:

- long-only weights
- weights sum to 1
- maximum weight per ticker

The implementation uses deterministic projected-gradient optimization and
does not require an external optimizer dependency.

## Leakage contract

The optimizer accepts only:

- ticker identifiers
- expected returns
- covariance matrix
- optimization configuration

There is deliberately no OOS-return input. Statistics must therefore be
prepared from a training/estimation window by the caller.

The optimized weights can subsequently be evaluated by Phase 19 and Phase 20,
but OOS outcomes must not be fed back into this optimizer for retuning.

## Scope

This phase is a research optimizer, not a live allocator. It does not perform
dynamic intraday optimization, shorting, leverage, transaction-cost-aware
optimization, or online learning. Those constraints can be introduced later
after the execution and risk contracts are frozen.
