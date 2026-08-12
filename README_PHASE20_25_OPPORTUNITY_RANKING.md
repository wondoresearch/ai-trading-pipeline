# Opportunity Ranking Architecture — Phase 20–25

## Purpose

This package continues the new research architecture after the new Phase 19
forward-return opportunity layer.

The objective is **not live trading**.

The objective is:

> Given any user-supplied stock universe, identify stocks with the best
> forward gain potential using sentiment + historical information, while
> explicitly accounting for risk.

## Architecture

User stock list
→ data sufficiency validation
→ forward-return prediction (Phase 19)
→ historical risk estimation
→ transparent risk-adjusted opportunity score
→ ranking

## New components

- `app/stock_universe.py`
  - accepts any ticker list
  - normalizes case/whitespace
  - rejects duplicates and malformed symbols
  - contains no hard-coded universe

- `app/universe_data_validation.py`
  - checks historical-data sufficiency
  - marks unavailable/insufficient stocks ineligible instead of silently
    fabricating data

- `app/risk_model.py`
  - annualized historical volatility
  - downside deviation
  - maximum drawdown
  - optional market beta

- `app/opportunity_score.py`
  - transparent expected-return/confidence/risk score
  - deliberately not a live-trading signal

- `app/opportunity_ranking.py`
  - ranks the complete user-defined universe

- `app/opportunity_pipeline.py`
  - orchestration layer

## Phase mapping

The repository already contains legacy Phase 19–21 portfolio/trading-oriented
artifacts. Do not overwrite those historical tags.

This package is the migration path for the new objective:

- Phase 20: user-defined universe + data eligibility
- Phase 21: historical risk model
- Phase 22: risk-adjusted opportunity score
- Phase 23: opportunity ranking
- Phase 24: end-to-end opportunity pipeline
- Phase 25: production-quality integration, reporting and final validation

The existing new Phase 19 remains the forward-return prediction and
cross-sectional evaluation layer.

## Leakage boundary

Historical risk inputs must be point-in-time data available before the
prediction timestamp.

Do not pass `t1_return`, `t3_return`, `t5_return`, `t10_return`, or any other
future outcome into the risk model as a feature.

## Applying

Copy the `app/` and `tests/` files into the repository.

Then run:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The package is intentionally additive and does not modify existing modules.
