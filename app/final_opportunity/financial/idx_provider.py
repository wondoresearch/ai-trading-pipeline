"""Robust public IDX financial-data adapter.

Handles the IDX public Digital Statistic page when the requested month has
real data, and explicitly detects the IDX "Data tidak tersedia" placeholder.
"""

from __future__ import annotations
import base64, json, re
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd
import requests

IDX_URL = (
    "https://www.idx.id/id/data-pasar/laporan-statistik/"
    "digital-statistic/monthly/financial-report-and-ratio-of-listed-companies/"
    "financial-data-and-ratio"
)

def _filter(year: int, month: int) -> str:
    payload = {"year": str(year), "month": str(month), "quarter": 0, "type": "monthly"}
    return base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode()

def _num(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().replace(" ", "")
    if s.lower() in {"", "-", "--", "nan", "none", "data tidak tersedia"}:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    s = re.sub(r"[^0-9+\-.]", "", s)
    try:
        return float(s)
    except ValueError:
        return None

def _flatten_columns(df):
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            vals = [str(x).strip() for x in c if str(x).strip().lower() != "nan"]
            cols.append(" | ".join(vals))
        else:
            cols.append(str(c).strip())
    df = df.copy()
    df.columns = cols
    return df

def _find_col(df, candidates):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    for c in df.columns:
        lc = str(c).strip().lower()
        if any(candidate.lower() in lc for candidate in candidates):
            return c
    return None

def _clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()

class IDXFinancialDataProvider:
    name = "idx_public_financial_data"
    source_url = IDX_URL

    def __init__(self, cache_dir="data/financial_idx", timeout=30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def _url(self, year, month):
        return f"{IDX_URL}?filter={_filter(year, month)}"

    def fetch_month(self, year: int, month: int, tickers: Optional[Iterable[str]] = None):
        url = self._url(year, month)
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 ai-trading-news-pipeline/1.0"},
            timeout=self.timeout,
        )
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        if not tables:
            return []

        target = _flatten_columns(tables[0])

        # IDX returns a one-row placeholder for months without publication.
        joined = " ".join(
            _clean(x).lower() for x in target.astype(object).iloc[0].tolist()
        ) if len(target) else ""
        if "data tidak tersedia" in joined:
            return []

        code_col = _find_col(target, ["Code"])
        sector_col = _find_col(target, ["Sector"])
        if code_col is None or sector_col is None:
            raise RuntimeError("IDX financial table layout is not recognized")

        wanted = {x.upper().replace(".JK", "") for x in tickers} if tickers else None
        out = []

        from .models import FinancialObservation

        for _, row in target.iterrows():
            ticker = _clean(row[code_col]).upper()
            if not ticker or ticker == "NAN" or (wanted and ticker not in wanted):
                continue

            def get(*names):
                col = _find_col(target, names)
                return row[col] if col is not None else None

            out.append(FinancialObservation(
                ticker=ticker,
                sector=_clean(get("Sector")),
                sub_industry=_clean(get("Sub Industry")),
                sub_industry_code=_clean(get("Sub Industry Code")),
                fs_date=_clean(get("FS Date")),
                fiscal_year_end=_clean(get("Fiscal Year End")),
                statement_type=_clean(get("Type of FS")),
                auditor_opinion=_clean(get("Auditor's Opinion")),
                assets=_num(get("Assets")),
                liabilities=_num(get("Liabilities")),
                equity=_num(get("Equity")),
                sales=_num(get("Sales")),
                ebt=_num(get("EBT")),
                profit=_num(get("Profit for the Period")),
                profit_attributed=_num(get("Profit attr.to owner's")),
                eps=_num(get("EPS")),
                book_value=_num(get("Book Value")),
                pe=_num(get("P/E Ratio")),
                pbv=_num(get("Price to BV")),
                debt_equity=_num(get("D/E Ratio")),
                roa=_num(get("ROA")),
                roe=_num(get("ROE")),
                npm=_num(get("NPM")),
            ))

        self.cache_dir.joinpath(f"{year:04d}-{month:02d}.json").write_text(
            json.dumps({
                "provider": self.name,
                "source_url": url,
                "retrieved_at_utc": datetime.utcnow().isoformat() + "Z",
                "rows": [x.to_dict() for x in out],
            }, indent=2),
            encoding="utf-8",
        )
        return out

    def latest_available(self, ticker: str, months_back: int = 18):
        today = date.today()
        wanted = ticker.upper().replace(".JK", "")
        for i in range(months_back):
            month = today.month - i
            year = today.year
            while month <= 0:
                month += 12
                year -= 1
            try:
                rows = self.fetch_month(year, month, [wanted])
                if rows:
                    return rows[0]
            except Exception:
                continue
        return None
