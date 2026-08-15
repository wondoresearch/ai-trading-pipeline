# Final Design Review — Phases 27–36

## Decision

Use one simple local research application:

- market data: Yahoo Finance through `yfinance` (no API key)
- news: Google News RSS
- storage: SQLite
- backend: FastAPI
- frontend: plain HTML/JavaScript
- scoring: existing transparent baseline until a validated Phase 12/13 model artifact is plugged in

## Critical gaps addressed

### 1. Paid-provider lock-in
The previous implementation required Twelve Data and failed for IDX symbols on the user's free plan. The final implementation removes that dependency entirely.

### 2. Ticker normalization
User-facing IDX tickers such as `BBRI` are normalized to Yahoo symbols such as `BBRI.JK` in exactly one adapter.

### 3. Data provenance
Every price row stores source, provider symbol, and retrieval time.

### 4. Point-in-time news gate
Ranking accepts an `as_of` cutoff and only uses news whose `available_at <= as_of`.

### 5. Future-adjusted price risk
Yahoo `Adj Close` can incorporate later corporate actions. It is therefore stored for reference but the baseline scoring path uses raw `Close`. Historical model validation must not treat current adjusted values as point-in-time observations.

### 6. Corporate actions
No silent total-return reconstruction is claimed. If a future validated model requires split/dividend-adjusted point-in-time prices, a dedicated corporate-action reconstruction phase must be added rather than leaking current adjustments backward.

### 7. Free universe discovery
There is no dependency on a paid universe endpoint. The user selects IDX tickers directly. The UI and CLI accept any Yahoo IDX symbol that resolves successfully.

### 8. News limitations
Google News RSS is a current research source, not a complete historical archive. Historical event-study artifacts already produced by the project remain separate.

### 9. Model duplication
No new ML model is introduced. The transparent baseline is explicitly labeled. Existing validated Phase 12/13 artifacts remain the preferred future producer.

### 10. Public/commercial use
`yfinance` documents that it is an unofficial open-source wrapper using Yahoo public APIs and that the Yahoo Finance API is intended for personal use. Therefore this free implementation is for personal/internal research only unless Yahoo's applicable terms permit the intended use.

## Final simple architecture

```text
User
  |
  v
Simple HTML/JS
  |
  v
FastAPI
  |
  v
ResearchService
  |
  +--> Yahoo Finance / yfinance --> IDX EOD history
  |
  +--> Google News RSS -----------> current news
  |
  v
SQLite
  |
  v
Transparent opportunity baseline
  |
  +--> expected return
  +--> probability gain
  +--> confidence
  +--> downside risk
  +--> data quality
  |
  v
Ranking + explainability
```

## Freeze criteria

1. No API key required for market data.
2. `BBRI` maps to `BBRI.JK`.
3. Real historical data can be synced.
4. News can be collected independently of market-data availability.
5. Missing data produces an explicit error/insufficient-evidence result rather than zero-filled fake evidence.
6. Ranking output is JSON-safe and rejects non-finite values.
7. `live_trading` remains false.
8. No broker/order/execution dependency is introduced.
9. Existing regression suite remains independent of this optional web layer.
10. Real-data validation must be performed locally before tagging a final freeze.
