# Phase 26 — Opportunity Runner / Integration Layer

## Purpose

Phase 26 makes the Phase 19 + Phase 20–25 opportunity engine runnable as a
research workflow.

This is **not live trading**.

It does not connect to a broker, place orders, or fabricate missing market or
news data.

## Current input contract

The runner accepts:

1. `--universe`: text file, one ticker per line.
2. `--prepared-data`: JSON containing point-in-time:
   - `predictions`
   - `confidence`
   - `historical_returns`
   - optional `market_returns`

Example:

```bash
python -m app.opportunity_runner \
  --universe data/example_universe.txt \
  --prepared-data data/example_prepared_data.json
```

Output:

```text
output/opportunity_ranking.json
```

## Why prepared data?

The existing repository has separate components for news collection,
sentiment, price data, event impact and forward-return evaluation. Phase 26
does not pretend that these provider/data boundaries are already integrated.

The runner therefore establishes a clean integration boundary first. The next
integration can replace `prepared-data` with adapters to the existing
point-in-time research pipeline without changing the ranking/risk contract.

## Output guarantees

The report is:

- JSON serializable
- `datetime` / `date` safe
- `None` for non-finite floating values
- `allow_nan=False`
- deterministic field insertion order
- explicitly marked `live_trading: false`

## Test

Run:

```bash
python -m unittest \
  tests.test_opportunity_runner \
  tests.test_opportunity_report
```

Then the complete suite:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## CLI

```bash
python -m app.opportunity_runner --help
```

## Important

`data/example_prepared_data.json` is only a deterministic smoke-test fixture.
It is not real market data and must not be interpreted as an investment result.
