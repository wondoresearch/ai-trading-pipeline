# Phase 18 — Stress Testing & Adversarial Robustness

Phase 18 evaluates frozen OOS strategy returns under controlled adverse
conditions. It is evaluation-only: stressed results must not be used to
retrain, retune thresholds, or select a model.

## Stress dimensions

- **Return shock** — scales realized strategy returns downward.
- **Signal noise** — deterministically flips a controlled fraction of signals.
- **Transaction cost** — subtracts a fixed cost per active signal.

Each scenario reports:

- sample size
- compounded total return
- mean return
- maximum drawdown
- degradation versus baseline
- relative robustness versus baseline when defined

## Contract

- input observations must be chronological
- no future information is introduced
- no model retraining
- no threshold tuning
- no selection based on stressed outcomes
- stress parameters are explicit and deterministic
- baseline is preserved as a separate reference

## Interpretation

Stress testing is a sensitivity analysis, not a guarantee of live
profitability. A strategy that survives moderate cost, return, and signal
perturbations with limited degradation has stronger practical robustness than
one whose performance collapses under small perturbations.

This phase does not establish statistical significance, realistic market
microstructure, or execution feasibility.
