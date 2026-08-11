# Phase 4 — Historical Price Observation Contract

Phase 4 resolves daily historical OHLCV observations from the Phase 3
`EventTimeResolution`.

Contract:
`effective_time -> previous trading-session baseline -> EVENT_DAY/T1/T3/T5/T10 observations`

Canonical downstream price is `Adj Close`. Raw `Close` is retained for auditability.

Files:
- `app/price_observation.py`
- `app/price_provider.py` (provider status boundary only)
- `tests/test_price_observation.py`

Phase 4 does not calculate returns or statistical significance.

Run:
`python -m unittest tests/test_price_observation.py -v`
