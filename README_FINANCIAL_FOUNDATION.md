# Financial Data Foundation — reviewed design

## Goal

Add financial-statement/fundamental information to the existing market/news
opportunity pipeline without making the system bank-specific.

## Free source decision

The primary source is the public **IDX Digital Statistic — Financial Data and
Ratio of Listed Companies** publication. IDX publicly exposes the publication
with fields including sector, sub-industry, assets, liabilities, equity, sales,
EBT, profit, EPS, book value, P/E, P/BV, D/E, ROA, ROE and NPM.

The implementation intentionally does **not** depend on a paid IDX Data Service.
IDX separately documents licensed IDX Market Data/Data Reference products, so
the code keeps this public-publication adapter isolated from any future paid
provider.

Source:
https://www.idx.id/id/data-pasar/laporan-statistik/digital-statistic/monthly/financial-report-and-ratio-of-listed-companies/financial-data-and-ratio

## Design review conclusions

1. **Sector neutral:** sector/sub-industry is carried with every observation.
2. **Financial vs non-financial:** banks/financial companies do not use
   industrial debt interpretation. Resource, industrial, consumer, technology,
   property, infrastructure and healthcare remain supported.
3. **No hard dependency:** fundamentals are an optional overlay. Existing
   market/news scoring continues to work if financial data is unavailable.
4. **Point-in-time discipline:** observations are identified by FS date and are
   fetched from a dated publication. Do not use a later restatement to
   backfill a historical event study without recording the retrieval vintage.
5. **Data quality:** source, retrieval timestamp, sector, statement type and
   auditor opinion are retained.
6. **No fabricated metrics:** missing financial fields remain null.
7. **Valuation caution:** P/E and P/BV are used only when positive and
   available; negative earnings are not converted into a fake valuation.
8. **Bounded influence:** the fundamental overlay is capped at ±0.20 so it
   cannot overwhelm price/news/event signals.
9. **Backtesting:** financial features must be joined using the latest report
   available *as of the event date*, not by today's latest report.
10. **Source resilience:** HTML parsing is isolated in `idx_provider.py`; a
    future XBRL/paid provider can implement the same normalized model without
    changing scoring.

## Important limitation

The public IDX publication is the preferred free source for the first
implementation, but the website is not an API contract. Layout changes can
break HTML parsing. The adapter therefore snapshots normalized observations
under `data/financial_idx/` and fails loudly when the expected table is absent.

For a production-grade historical backtest, add a versioned raw-document
archive or a licensed/XBRL source. Do not silently mix vintages.

## First run

Install dependencies if needed:

    pip install pandas requests lxml

Fetch a month:

    PYTHONPATH=. python scripts/fetch_idx_financials.py --year 2025 --month 9 --tickers BBRI BBCA BMRI BBNI TLKM

Check the latest public observation found:

    PYTHONPATH=. python scripts/financial_status.py --ticker BBRI

Run tests:

    PYTHONPATH=. python -m unittest tests.test_financial_features -v

## Integration rule

Do not replace the existing opportunity score. Use
`financial.enrichment.fundamental_adjustment()` as an optional additive
feature. The next phase should join it to the event-time engine using
`fs_date <= event_time` and the report publication/availability timestamp,
then evaluate incremental predictive value out-of-sample.
