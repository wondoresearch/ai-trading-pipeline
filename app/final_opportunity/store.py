from __future__ import annotations
import json, sqlite3
from pathlib import Path
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS universe (
  ticker TEXT PRIMARY KEY,
  name TEXT,
  exchange TEXT,
  provider_symbol TEXT,
  currency TEXT,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prices (
  ticker TEXT NOT NULL,
  trading_date TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume REAL,
  source TEXT NOT NULL,
  provider_symbol TEXT,
  retrieved_at TEXT NOT NULL,
  PRIMARY KEY (ticker, trading_date, source)
);
CREATE TABLE IF NOT EXISTS news (
  news_id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  title TEXT NOT NULL,
  url TEXT,
  source TEXT,
  published_at TEXT NOT NULL,
  available_at TEXT NOT NULL,
  sentiment REAL,
  sentiment_label TEXT,
  sentiment_source TEXT
);
CREATE TABLE IF NOT EXISTS rankings (
  as_of TEXT NOT NULL,
  ticker TEXT NOT NULL,
  horizon_days INTEGER NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (as_of, ticker, horizon_days)
);
"""

class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.executescript(SCHEMA)
            # Safe additive migration for databases created by the first ZIP.
            cols = {r[1] for r in db.execute("PRAGMA table_info(prices)")}
            if "adj_close" not in cols:
                db.execute("ALTER TABLE prices ADD COLUMN adj_close REAL")
            if "provider_symbol" not in cols:
                db.execute("ALTER TABLE prices ADD COLUMN provider_symbol TEXT")

    def upsert_universe(self, rows: Iterable[dict]) -> None:
        with sqlite3.connect(self.path) as db:
            db.executemany(
                """INSERT OR REPLACE INTO universe
                (ticker,name,exchange,provider_symbol,currency,updated_at)
                VALUES(?,?,?,?,?,?)""",
                [(r["ticker"], r.get("name"), r.get("exchange","IDX"),
                  r.get("provider_symbol"), r.get("currency","IDR"), r["updated_at"]) for r in rows],
            )

    def upsert_prices(self, rows: Iterable[dict]) -> None:
        with sqlite3.connect(self.path) as db:
            db.executemany(
                """INSERT OR REPLACE INTO prices
                (ticker,trading_date,open,high,low,close,adj_close,volume,source,provider_symbol,retrieved_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                [(r["ticker"],r["trading_date"],r.get("open"),r.get("high"),r.get("low"),
                  r.get("close"),r.get("adj_close"),r.get("volume"),r["source"],
                  r.get("provider_symbol"),r["retrieved_at"]) for r in rows],
            )

    def upsert_news(self, rows: Iterable[dict]) -> None:
        with sqlite3.connect(self.path) as db:
            db.executemany(
                """INSERT OR REPLACE INTO news
                (news_id,ticker,title,url,source,published_at,available_at,sentiment,sentiment_label,sentiment_source)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                [(r["news_id"],r["ticker"],r["title"],r.get("url"),r.get("source"),
                  r["published_at"],r["available_at"],r.get("sentiment"),r.get("sentiment_label"),
                  r.get("sentiment_source")) for r in rows],
            )

    def prices(self, ticker: str) -> list[dict]:
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            return [dict(x) for x in db.execute(
                "SELECT * FROM prices WHERE ticker=? ORDER BY trading_date", (ticker,)
            )]

    def news(self, ticker: str, limit: int = 50, cutoff: str | None = None) -> list[dict]:
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            if cutoff:
                return [dict(x) for x in db.execute(
                    "SELECT * FROM news WHERE ticker=? AND available_at<=? ORDER BY published_at DESC LIMIT ?",
                    (ticker, cutoff, limit)
                )]
            return [dict(x) for x in db.execute(
                "SELECT * FROM news WHERE ticker=? ORDER BY published_at DESC LIMIT ?", (ticker, limit)
            )]

    def universe(self) -> list[dict]:
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            return [dict(x) for x in db.execute("SELECT * FROM universe ORDER BY ticker")]

    def save_ranking(self, as_of: str, horizon: int, rows: list[dict]) -> None:
        with sqlite3.connect(self.path) as db:
            db.executemany(
                "INSERT OR REPLACE INTO rankings(as_of,ticker,horizon_days,payload_json) VALUES(?,?,?,?)",
                [(as_of,r["ticker"],horizon,json.dumps(r,sort_keys=True,allow_nan=False)) for r in rows],
            )
