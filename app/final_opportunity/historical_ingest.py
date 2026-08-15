"""Historical IDX EOD ingestion.

Accepts one CSV or a directory of CSV files, normalizes common IDX-style
columns into data/market_eod/TICKER.csv, removes duplicate dates, and writes
provenance metadata. No network access is performed by this module.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

COLUMN_ALIASES = {
    "ticker": ["StockCode", "stock_code", "Ticker", "ticker", "Code", "code"],
    "date": ["Date", "date", "TradeDate", "trade_date"],
    "open": ["OpenPrice", "open_price", "Open", "open"],
    "high": ["High", "high"],
    "low": ["Low", "low"],
    "close": ["Close", "close", "ClosePrice", "close_price"],
    "volume": ["Volume", "volume", "TradeVolume", "trade_volume"],
}

OUTPUT_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _pick(row: Dict[str, str], names: List[str]) -> Optional[str]:
    for name in names:
        if name in row:
            return row[name]
    return None


def _date(value: str) -> str:
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Unsupported date: {value}")


def _number(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip().replace(",", "")
    if value in ("", "-", "--", "null", "None"):
        return None
    return float(value)


def _files(source: str) -> Iterable[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(x for x in p.glob("*.csv") if x.is_file())
    raise FileNotFoundError(source)


def import_source(
    source: str,
    output_dir: str = "data/market_eod",
    tickers: Optional[Iterable[str]] = None,
    source_name: str = "external_historical_dataset",
    source_url: Optional[str] = None,
) -> Dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    allowed = {t.upper().replace(".JK", "") for t in tickers} if tickers else None

    buckets: Dict[str, Dict[str, Dict]] = {}
    input_files = 0
    input_rows = 0
    rejected_rows = 0

    for path in _files(source):
        input_files += 1
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                input_rows += 1
                ticker = _pick(raw, COLUMN_ALIASES["ticker"]) or path.stem
                try:
                    if not ticker:
                        raise ValueError("missing ticker")
                    ticker = ticker.strip().upper().replace(".JK", "")
                    if allowed and ticker not in allowed:
                        continue

                    row = {
                        "date": _date(_pick(raw, COLUMN_ALIASES["date"]) or ""),
                        "open": _number(_pick(raw, COLUMN_ALIASES["open"])),
                        "high": _number(_pick(raw, COLUMN_ALIASES["high"])),
                        "low": _number(_pick(raw, COLUMN_ALIASES["low"])),
                        "close": _number(_pick(raw, COLUMN_ALIASES["close"])),
                        "volume": _number(_pick(raw, COLUMN_ALIASES["volume"])),
                    }

                    if any(row[k] is None for k in ("open", "high", "low", "close")):
                        raise ValueError("missing OHLC")

                    if not (row["low"] <= row["open"] <= row["high"]):
                        raise ValueError("open outside low/high")
                    if not (row["low"] <= row["close"] <= row["high"]):
                        raise ValueError("close outside low/high")

                    buckets.setdefault(ticker, {})[row["date"]] = row
                except (ValueError, TypeError):
                    rejected_rows += 1

    written = 0
    coverage = {}

    for ticker, rows in sorted(buckets.items()):
        ordered = [rows[d] for d in sorted(rows)]
        target = out / f"{ticker}.csv"
        with target.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(ordered)

        written += len(ordered)
        coverage[ticker] = {
            "rows": len(ordered),
            "first_date": ordered[0]["date"],
            "last_date": ordered[-1]["date"],
        }

    manifest = {
        "source_name": source_name,
        "source_url": source_url,
        "source_path": str(Path(source).resolve()),
        "imported_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input_files": input_files,
        "input_rows": input_rows,
        "rejected_rows": rejected_rows,
        "tickers_written": len(coverage),
        "rows_written": written,
        "coverage": coverage,
    }
    (out / "_provenance.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
