# Phase 5 — Return Engine

Phase 5 converts a Phase 4 `PriceObservationSet` into simple returns without
fetching prices or changing event-time decisions.

Contract:
`event_day_return = EVENT_DAY adjusted close / previous-session adjusted close - 1`

`forward_return = Tn adjusted close / EVENT_DAY adjusted close - 1`

`cumulative_return = Tn adjusted close / previous-session adjusted close - 1`

The engine preserves the Phase 4 event, ticker, effective time, baseline date,
price source, and daily price granularity. It exposes an event-day result plus
`T1`, `T3`, `T5`, and `T10` forward observations.

Unavailable or invalid Phase 4 prices never produce a numeric return. Missing
baseline, missing event-day data, missing individual forward data, invalid
data, provider errors, and zero denominators are explicit in each affected
return and in the overall result.

Files:
- `app/return_engine.py`
- `tests/test_return_engine.py`

Run:
`python -m unittest tests/test_return_engine.py -v`
