# Market Contract Hardening V2

This patch closes the remaining market/store/event-time integration gaps in
`feat/opportunity-ranking-engine`.

## Changes

1. Canonical market provider output now matches `Store.upsert_prices()`.
2. `MarketDataProvider.stocks()` is defined and implemented by the local IDX provider.
3. Market observations carry `available_at` for historical event-time cutoffs.
4. `Store.prices()` filters by `available_at` rather than only `trading_date`.
5. `adj_close` remains optional and is never fabricated.
6. ResearchService uses the explicit `FreeIDXMarketData` provider.
7. ResearchService output provenance is no longer labeled Yahoo.
8. Regression tests cover provider -> store -> cutoff compatibility.
9. GitHub Actions runs the complete regression suite on push/PR.

## Event-time policy

The local IDX EOD importer treats an EOD observation as available at
16:30:00 Asia/Jakarta. This is a conservative project-level assumption and
should be replaced with an observed source timestamp if the upstream importer
can provide one.


## Patch 1

Fixes a regression in `Store.prices(cutoff=...)`: the cutoff now always enforces
`trading_date <= cutoff` and, when `available_at` is populated,
`available_at <= cutoff`. This prevents an observation whose market bar is in
the future from bypassing the trading-date PIT boundary merely because its
availability timestamp is earlier.
