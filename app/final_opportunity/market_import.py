"""Import official/community IDX EOD exports into normalized per-ticker CSV files."""

from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ALIASES = {
    "ticker": {"ticker", "stockcode", "stock_code", "kode", "kode saham", "stock"},
    "date": {"date", "tanggal", "tradedate", "trade_date"},
    "open": {"open", "openprice", "open_price", "harga pembukaan"},
    "high": {"high", "highprice", "high_price", "harga tertinggi"},
    "low": {"low", "lowprice", "low_price", "harga terendah"},
    "close": {"close", "closeprice", "close_price", "harga penutupan"},
    "volume": {"volume", "volume saham", "sharevolume", "tradeablevolume"},
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def _find(headers: Iterable[str], logical: str) -> str | None:
    wanted = {_norm(x) for x in ALIASES[logical]}
    for h in headers:
        if _norm(h) in wanted:
            return h
    return None


def _read_csv_bytes(data: bytes) -> List[Dict]:
    text = data.decode("utf-8-sig", errors="replace")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    headers = reader.fieldnames or []

    mapping = {logical: _find(headers, logical) for logical in ALIASES}
    missing = [k for k in ("ticker", "date", "close") if not mapping[k]]
    if missing:
        return []

    output = []
    for row in reader:
        ticker = str(row[mapping["ticker"]]).strip().upper()
        ticker = ticker.replace(".JK", "")
        if not re.fullmatch(r"[A-Z0-9]+", ticker):
            continue
        output.append({
            "ticker": ticker,
            "date": str(row[mapping["date"]]).strip(),
            "open": str(row[mapping["open"]]).strip() if mapping["open"] else "",
            "high": str(row[mapping["high"]]).strip() if mapping["high"] else "",
            "low": str(row[mapping["low"]]).strip() if mapping["low"] else "",
            "close": str(row[mapping["close"]]).strip(),
            "volume": str(row[mapping["volume"]]).strip() if mapping["volume"] else "",
        })
    return output


def _files(source: Path) -> Iterable[Tuple[str, bytes]]:
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            for name in zf.namelist():
                if name.lower().endswith((".csv", ".txt")):
                    yield name, zf.read(name)
    else:
        yield source.name, source.read_bytes()


def import_market_file(source: str, output_dir: str = "data/market_eod") -> Dict:
    source_path = Path(source)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, Dict[str, Dict]] = {}
    files_read = 0
    rows_read = 0

    for name, payload in _files(source_path):
        files_read += 1
        for row in _read_csv_bytes(payload):
            rows_read += 1
            grouped.setdefault(row["ticker"], {})[row["date"]] = row

    written = 0
    for ticker, by_date in grouped.items():
        path = out / f"{ticker}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["date", "open", "high", "low", "close", "volume"],
            )
            writer.writeheader()
            for d in sorted(by_date):
                row = by_date[d]
                writer.writerow({k: row[k] for k in writer.fieldnames})
        written += 1

    return {
        "source": str(source_path),
        "files_read": files_read,
        "rows_read": rows_read,
        "tickers_written": written,
        "output_dir": str(out),
    }
