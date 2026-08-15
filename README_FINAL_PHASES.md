# Final Opportunity Research System — Design + Implementation

## Objective

Given a user-defined stock universe, identify stocks with the strongest
risk-adjusted, evidence-backed potential return over a selected future
horizon.

This is a research decision-support system. It is NOT:
- live trading
- order execution
- broker integration
- portfolio allocation
- position sizing
- BUY/SELL instruction generation

## Design review — critical gaps found and fixed

### 1. Prediction provenance gap
Phase 26 accepted `prepared_data.json`, so expected return had no real producer.
This package makes the producer explicit: a transparent baseline derived from
real historical prices + recent news sentiment. A future validated ML artifact
can replace the baseline without changing the web/data contract.

### 2. Point-in-time leakage
Every news row has `published_at` and `available_at`. The market data is stored
with retrieval provenance. Future realized returns are not used by the current
ranking path.

### 3. Adjusted-price ambiguity
The final baseline uses `adjust=none` from Yahoo Finance. It does NOT silently
mix future-adjusted prices into point-in-time prediction. Corporate-action/
total-return modeling can be added separately after a point-in-time action feed
is available.

### 4. Survivorship bias
The UI accepts a user-defined current universe. Historical validation must not
reuse today's universe; it must use a point-in-time universe snapshot. The
database therefore stores universe rows separately from predictions.

### 5. News coverage bias
RSS is not treated as a complete historical news database. It is a current
research source. Historical event-impact data already produced by the existing
pipeline remains a separate research artifact and should not be silently mixed
with current news.

### 6. Ticker/entity ambiguity
The first implementation queries the user's ticker and, when available, the
provider company name. It does not perform fuzzy ticker inference. This avoids
false positives at the cost of lower recall.

### 7. Risk gate
A stock with insufficient price history or insufficient news evidence is
excluded from strong-opportunity status instead of being filled with zeros.

### 8. Black-box score
The score is decomposable:
- expected return
- probability of gain
- confidence
- downside risk
- data quality

The UI exposes these components.

### 9. Provider dependency
Yahoo Finance is isolated behind one adapter. The rest of the system does not
depend on provider-specific response formats.

### 10. Frontend/backend coupling
The web UI talks only to FastAPI JSON endpoints. The research engine remains
usable from CLI without the browser.

## Simple final architecture

```text
User
  |
  v
FastAPI + simple HTML/JS
  |
  v
ResearchService
  |
  +--> Yahoo Finance --> EOD OHLCV
  |
  +--> Google News RSS --> current news
  |
  v
SQLite research store
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
Risk-adjusted ranking
```

SQLite is intentionally used instead of introducing DuckDB/PostgreSQL/Redis.
For this research workload it is enough and keeps the deployment simple.

## Real data

Yahoo Finance via `yfinance` is the primary market provider because it requires no API key and exposes IDX symbols using the `.JK` suffix (for example `BBRI.JK`). The adapter stores both raw Close and Adjusted Close. Adjusted Close is used only for return calculations so dividend/split effects are not silently treated as price moves.

This source is intentionally limited to personal research/education. yfinance documents that it is an unofficial open-source wrapper around Yahoo public APIs and that the Yahoo Finance API is intended for personal use. Do not expose or redistribute the raw Yahoo data through a public/commercial service without checking Yahoo's terms.

No market-data API key is required.

## News

The first real source is Google News RSS search per selected ticker/company.
This is deliberately treated as a current-news source, not a complete
historical archive.

The existing project sentiment pipeline remains the preferred NLP layer when
it is already integrated. This package uses a tiny transparent lexicon baseline
only to make the final system runnable without adding another heavy ML
dependency. Its output is labeled `lexicon_baseline_v1`.

## Why not use a new ML model here?

The existing Phase 10–18 pipeline already contains feature engineering,
controlled training, OOS validation and robustness testing. Re-training or
introducing a second production model in the final UI phase would create a
second source of truth.

The final system therefore starts with a transparent baseline. A validated
Phase 12/13 model artifact can later be plugged in behind the same prediction
contract.

## CLI

Install additive dependencies:

```bash
pip install -r requirements-phase27-final.txt
```

Configure:

```bash
```

Sync real data:

```bash
python -m app.final_opportunity.cli sync --tickers BBRI BBCA BMRI BBNI
```

Analyze:

```bash
python -m app.final_opportunity.cli analyze \
  --tickers BBRI BBCA BMRI BBNI \
  --horizon 20 \
  --refresh
```

Run web UI:

```bash
python -m app.final_opportunity.cli serve
```

Open:

```text
http://127.0.0.1:8000
```

## Final phase plan

27. Free real market-data adapter + IDX ticker normalization
28. Local historical data store + adjusted-close provenance
29. Historical backfill/incremental sync
30. Real news RSS ingestion
31. Point-in-time research snapshot + leakage gates
32. FastAPI research API
33. Simple web UI
34. Explainability/data provenance
35. Scheduled refresh (optional OS cron)
36. End-to-end validation/freeze

All are included in this package as one additive implementation. The phases
are logical boundaries, not separate code branches.

## Freeze rule

Before freezing:
1. run the new tests
2. run the full existing regression suite
3. run `analyze` against a real configured universe
4. verify output contains no NaN/Infinity
5. verify `live_trading=false`
6. verify no execution/broker imports
7. verify the same input produces the same ranking when data snapshot is held
constant

## Important methodological limitation

The baseline prediction is NOT a claim of statistically validated future
performance. It is a transparent first real-data ranking signal. The existing
Phase 19–25 OOS validation should be used to validate any replacement model or
formula before treating its ranking as reliable.

## Free-data decision

This implementation intentionally does not require Twelve Data or another paid market-data API. Yahoo Finance via yfinance is used because IDX symbols such as BBRI.JK have daily historical data available on Yahoo Finance. yfinance is an unofficial open-source wrapper and its own documentation states that the Yahoo Finance API is intended for personal use; this project therefore treats the source as personal/internal research data, not a source for commercial redistribution.

For a public/commercial deployment, replace the `YahooFinanceMarketData` adapter with a properly licensed provider without changing the Store, scoring, API, or frontend contracts.
