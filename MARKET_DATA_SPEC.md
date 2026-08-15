# Market Data Contract — Phase 27–36

## Canonical storage

`data/market_eod/<TICKER>.csv`

Required:

`date,open,high,low,close,volume`

Optional:

`adj_close`

## Validation rules

1. One row per trading date.
2. Dates must be parseable and must not be in the future.
3. Duplicate dates are invalid.
4. OHLC must satisfy:
   - high >= low
   - high >= open and close
   - low <= open and close
   - close > 0
5. Volume must be >= 0.
6. Missing OHLCV is not silently filled.
7. Corporate-action adjustment is not inferred by this layer. If adjusted prices are required, the source must explicitly provide/document the adjustment methodology.
8. Source acquisition and normalization remain separate so the market source can change without changing the research engine.

## Research limitation

"Free" does not mean "authoritative". A public/community dataset can be used for research, but its provenance and coverage must be recorded. The normalized store must not silently mix data from different sources.
