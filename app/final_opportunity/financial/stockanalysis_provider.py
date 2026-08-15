"""Robust public StockAnalysis financial-data adapter.

The parser is deliberately tolerant of the HTML table shapes used by
StockAnalysis. It supports both simple tables and the MultiIndex-style tables
returned by pandas.read_html from the live site.

This is a public-web adapter, not a licensed API.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Optional
import json
import re

import pandas as pd
import requests


BASE_URL = "https://stockanalysis.com/quote/idx/{ticker}/financials/"


@dataclass(frozen=True)
class FinancialRecord:
    ticker: str
    period: str
    period_end: Optional[str]
    period_type: str
    currency: Optional[str]
    unit: Optional[str]
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    pretax_income: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None
    source: str = "stockanalysis_web"
    source_url: str = ""
    retrieved_at_utc: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _flat(value: Any) -> str:
    if isinstance(value, tuple):
        parts = [str(x).strip() for x in value if str(x).strip() and str(x).lower() != "nan"]
        return " | ".join(parts)
    return str(value).strip()


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", _flat(value)).strip().lower()


def _parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "n/a", "na", "-", "—", "--"}:
        return None
    s = s.replace(",", "").replace("%", "").strip()
    suffix = s[-1:].upper()
    multiplier = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}.get(suffix, 1.0)
    if multiplier != 1.0:
        s = s[:-1]
    try:
        return float(s) * multiplier
    except ValueError:
        return None


def _currency_unit(text: str) -> tuple[Optional[str], Optional[str]]:
    low = text.lower()
    currency = None
    m = re.search(r"financials\s+in\s+(?:millions|billions|trillions|thousands)\s+([a-z]{3})", low)
    if m:
        currency = m.group(1).upper()
    unit = None
    for word in ("trillions", "billions", "millions", "thousands"):
        if word in low:
            unit = word
            break
    if currency is None:
        m = re.search(r"\b([A-Z]{3})\b", text)
        if m:
            currency = m.group(1).upper()
    return currency, unit


_METRICS = {
    "revenue": "revenue",
    "gross profit": "gross_profit",
    "operating income": "operating_income",
    "pretax income": "pretax_income",
    "net income": "net_income",
    "earnings per share": "eps",
    "eps": "eps",
}


def _metric_from_label(label: Any) -> Optional[str]:
    s = _norm(label)
    # Actual StockAnalysis tables can expose labels such as
    # "Revenue Revenue Growth" and "Net Income Net Income Growth".
    for needle, key in sorted(_METRICS.items(), key=lambda x: -len(x[0])):
        if s == needle or s.startswith(needle + " "):
            if "growth" in s and s.startswith(needle + " growth"):
                return None
            return key
    return None


def _period_columns(df: pd.DataFrame) -> list[tuple[int, str]]:
    cols = list(df.columns)
    result = []
    for i, c in enumerate(cols):
        s = _flat(c)
        if re.search(r"\b(?:FY\s+\d{4}|TTM|Q[1-4]\s+\d{4})\b", s, re.I):
            result.append((i, s))
    return result


def _period_end(period: str) -> Optional[str]:
    m = re.search(r"FY\s+(\d{4})", period, re.I)
    if m:
        return f"{m.group(1)}-12-31"
    m = re.search(r"(?:TTM|Q[1-4])\s+(\d{4})", period, re.I)
    if m:
        return f"{m.group(1)}-12-31"
    return None


def _find_income_table(tables: list[pd.DataFrame]) -> Optional[pd.DataFrame]:
    for df in tables:
        if df.empty:
            continue
        text = " ".join(_norm(x) for x in df.columns)
        text += " " + " ".join(_norm(x) for x in df.iloc[:25, :].to_numpy().ravel())
        if "revenue" in text and "net income" in text:
            return df.copy()
    return None


def _normalize_income_table(
    df: pd.DataFrame,
    ticker: str,
    period_type: str,
    currency: Optional[str],
    unit: Optional[str],
    source_url: str,
    retrieved_at: str,
) -> list[FinancialRecord]:
    period_cols = _period_columns(df)
    if not period_cols:
        # Simple fixtures often have the periods as ordinary string columns.
        period_cols = [
            (i, _flat(c)) for i, c in enumerate(df.columns)
            if re.search(r"\b(?:FY\s+\d{4}|TTM|Q[1-4]\s+\d{4})\b", _flat(c), re.I)
        ]
    if not period_cols:
        return []

    metric_rows: dict[str, list[Any]] = {}
    for _, row in df.iterrows():
        label = _metric_from_label(row.iloc[0])
        if label:
            metric_rows[label] = list(row)

    # Some pandas versions/index configurations make the row label the index.
    if not metric_rows:
        for idx, row in df.iterrows():
            label = _metric_from_label(idx)
            if label:
                metric_rows[label] = list(row)

    if not metric_rows:
        return []

    records = []
    for pos, period in period_cols:
        def val(key: str) -> Optional[float]:
            values = metric_rows.get(key)
            if values is None or pos >= len(values):
                return None
            return _parse_number(values[pos])

        records.append(
            FinancialRecord(
                ticker=ticker,
                period=period,
                period_end=_period_end(period),
                period_type=period_type,
                currency=currency,
                unit=unit,
                revenue=val("revenue"),
                gross_profit=val("gross_profit"),
                operating_income=val("operating_income"),
                pretax_income=val("pretax_income"),
                net_income=val("net_income"),
                eps=val("eps"),
                source_url=source_url,
                retrieved_at_utc=retrieved_at,
            )
        )
    return records


class StockAnalysisFinancialProvider:
    name = "stockanalysis_web"
    source_type = "public_web"

    def __init__(
        self,
        cache_dir: str = "data/financial_stockanalysis",
        timeout: int = 30,
        session: Optional[requests.Session] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 Chrome/151.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

    def url(self, ticker: str, quarterly: bool = False) -> str:
        ticker = ticker.upper().replace(".JK", "")
        return BASE_URL.format(ticker=ticker) + ("?p=quarterly" if quarterly else "")

    def fetch(self, ticker: str, quarterly: bool = False) -> list[FinancialRecord]:
        ticker = ticker.upper().replace(".JK", "")
        url = self.url(ticker, quarterly)
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        body = response.text
        if "does not exist" in body.lower():
            return []

        tables = pd.read_html(StringIO(body))
        table = _find_income_table(tables)
        if table is None:
            return []

        currency, unit = _currency_unit(body)
        retrieved = datetime.now(timezone.utc).isoformat()
        records = _normalize_income_table(
            table, ticker, "quarterly" if quarterly else "annual",
            currency, unit, url, retrieved
        )
        if records:
            cache = self.cache_dir / f"{ticker}_{'quarterly' if quarterly else 'annual'}.json"
            cache.write_text(json.dumps([r.to_dict() for r in records], indent=2), encoding="utf-8")
        return records

    def latest(self, ticker: str) -> Optional[FinancialRecord]:
        annual = self.fetch(ticker, quarterly=False)
        if annual:
            return annual[0]
        quarterly = self.fetch(ticker, quarterly=True)
        return quarterly[0] if quarterly else None

    def health(self) -> dict:
        return {
            "provider": self.name,
            "source_type": self.source_type,
            "available": True,
            "licensed_api": False,
        }


def provider_smoke_test(tickers: Iterable[str] = ("BBRI", "BBCA", "PTBA", "ADRO")) -> dict:
    provider = StockAnalysisFinancialProvider()
    result = {}
    for ticker in tickers:
        try:
            rows = provider.fetch(ticker)
            result[ticker] = {
                "ok": bool(rows),
                "rows": len(rows),
                "sample": rows[0].to_dict() if rows else None,
            }
        except Exception as exc:
            result[ticker] = {
                "ok": False,
                "rows": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return result
