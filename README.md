# AI Trading News Pipeline

Prototype pipeline:

RSS / News Feed -> Collector -> Deduplication -> Entity Extraction -> Sentiment -> Ticker Mapping -> JSON

## Requirements
- Python 3.10+ recommended
- Internet access for RSS feeds and first-time Hugging Face model download

## Setup

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want custom settings.

## Run once

```bash
python main.py
```

The result is written to `output/news_result.json`.

## Run continuously

```bash
python worker.py
```

The worker polls the RSS feeds at the configured interval.

## Add IDX companies

Edit:

`data/ticker_master.json`

The prototype uses alias matching. For production, replace it with a proper IDX master and later add NER/entity linking.

## Notes

This is an NLP/data-ingestion prototype, not a trading recommendation system. The sentiment model is a general Indonesian sentiment model and is NOT trained specifically for financial markets. Before using it for trading, validate it against labeled financial-news data and perform time-series backtesting.
