# Phase 27–36 — Free IDX Market Data Layer

## Decision

Yahoo Finance is removed from the primary market-data path. The project uses **local normalized EOD IDX data** as the market-data source of truth.

This avoids:
- Yahoo rate limiting (HTTP 429)
- paid API keys
- provider-specific ticker suffixes
- unstable unofficial APIs

The official IDX website publishes daily statistical reports. The project intentionally expects the user to download the permitted daily EOD file and import it rather than scraping IDX. IDX states that it provides EOD market data products, and its website terms prohibit web scraping/crawling.

## Flow

Official IDX EOD export
        ↓
`import-market`
        ↓
`data/market_eod/*.csv`
        ↓
`FreeIDXMarketData`
        ↓
SQLite `prices`
        ↓
opportunity scoring/ranking

## Supported input

CSV or ZIP containing CSV/TXT rows with these logical fields:

- date
- ticker / StockCode
- open / OpenPrice
- high / High
- low / Low
- close / Close
- volume / Volume

The importer accepts common IDX naming variants and normalizes them.

## Commands

Import an IDX daily file:

```bash
python -m app.final_opportunity.cli import-market --file /path/to/IDX_EOD.zip
```

or:

```bash
python -m app.final_opportunity.cli import-market --file /path/to/stock_summary.csv
```

Then sync:

```bash
python -m app.final_opportunity.cli sync --tickers BBRI BBCA BMRI BBNI TLKM
```

The market layer will use local EOD data. News continues to use Google News RSS.

## Important data policy

The project does not scrape IDX. The official IDX site states that non-commercial quotation is permitted with attribution and date of access, while commercial use/distribution requires permission. Verify the applicable IDX terms for your use case.

For research, a community-maintained IDX dataset can be used as a historical bootstrap, but its repository states that the data originates from IDX and is CC BY-NC 4.0. Do not use it commercially without appropriate permission.

## No new Python dependency

This implementation uses only Python standard library for market-data ingestion. Existing project dependencies remain unchanged.


## One-shot apply

From the target repository:

```bash
unzip -q ~/Downloads/ai_trading_opportunity_free_idx_market_phase27_36.zip -d /tmp/free-idx-market
bash /tmp/free-idx-market/ai_trading_opportunity_free_idx_market_phase27_36/scripts/apply_free_idx_market.sh "$PWD"
```

Then run:

```bash
python -m unittest tests.test_market_free -v
```

## Getting data

Use an IDX EOD export that you are permitted to download. The importer accepts a CSV directly or a ZIP containing CSV/TXT files.

Example:

```bash
python -m app.final_opportunity.cli import-market   --file ~/Downloads/IDX_EOD.zip
```

Verify:

```bash
find data/market_eod -maxdepth 1 -name '*.csv' | head
```

Then:

```bash
python -m app.final_opportunity.cli sync   --tickers BBRI BBCA BMRI BBNI TLKM
```

If a requested ticker has no imported file, sync reports a market-data error instead of silently treating missing prices as zero.

## Design choice

This phase intentionally does **not** add a second unofficial web scraper. Current evidence shows the official IDX website provides daily statistical reports and EOD data products, while its terms prohibit web scraping/crawling. The free layer therefore uses user-downloaded EOD exports as the ingestion boundary. citeturn1view1turn0search4
