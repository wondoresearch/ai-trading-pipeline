"""Free Yahoo Finance financial-data provider.

Uses the public Yahoo Finance data exposed through the `yfinance` package.
This is an unofficial/community client, so the provider is deliberately
isolated behind the FinancialDataProvider-style interface.

Important:
- Yahoo provides period-end financial statements, but not a reliable filing
  publication timestamp for every observation.
- Therefore publication_date remains None and these observations are NOT
  event-time eligible unless a separate publication source supplies it.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import FinancialObservation


class YahooFinanceFinancialProvider:
    name = "yahoo_finance_public"
    source_url = "https://finance.yahoo.com"

    def __init__(self, cache_dir: str = "data/financial_yahoo", timeout: int = 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    @staticmethod
    def _symbol(ticker: str) -> str:
        t = ticker.upper().strip()
        return t if "." in t else f"{t}.JK"

    @staticmethod
    def _num(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pick(df: pd.DataFrame, *names):
        if df is None or df.empty:
            return None
        normalized = {str(i).strip().lower(): i for i in df.index}
        for name in names:
            key = name.lower()
            if key in normalized:
                row = df.loc[normalized[key]]
                return row
        for idx in df.index:
            text = str(idx).strip().lower()
            for name in names:
                if name.lower() in text:
                    return df.loc[idx]
        return None

    @staticmethod
    def _latest_value(df: pd.DataFrame, *names):
        row = YahooFinanceFinancialProvider._pick(df, *names)
        if row is None:
            return None
        try:
            s = row.dropna()
            return YahooFinanceFinancialProvider._num(s.iloc[0]) if not s.empty else None
        except Exception:
            return None

    @staticmethod
    def _latest_period(df: pd.DataFrame):
        if df is None or df.empty:
            return None
        try:
            return pd.Timestamp(df.columns[0]).date().isoformat()
        except Exception:
            return None

    def fetch(self, ticker: str, quarterly: bool = False) -> Optional[FinancialObservation]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "yfinance is required for YahooFinanceFinancialProvider. "
                "Install with: pip install yfinance"
            ) from exc

        symbol = self._symbol(ticker)
        tk = yf.Ticker(symbol)

        income = tk.quarterly_income_stmt if quarterly else tk.income_stmt
        balance = tk.quarterly_balance_sheet if quarterly else tk.balance_sheet

        if income is None or income.empty:
            return None

        period = self._latest_period(income)
        if not period:
            return None

        # Yahoo uses "Total Revenue" for non-bank companies and also exposes
        # bank revenue-like totals. The raw values are kept as reported.
        revenue = self._latest_value(income, "Total Revenue", "Operating Revenue")
        ebt = self._latest_value(
            income,
            "Pretax Income",
            "PretaxIncome",
        )
        profit = self._latest_value(
            income,
            "Net Income",
            "Net Income Common Stockholders",
        )
        eps = self._latest_value(
            income,
            "Diluted EPS",
            "Basic EPS",
        )

        assets = self._latest_value(balance, "Total Assets")
        liabilities = self._latest_value(balance, "Total Liabilities Net Minority Interest")
        equity = self._latest_value(
            balance,
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest",
        )

        # Profile metadata is optional; failures here must not invalidate
        # financial statements.
        sector = ""
        sub_industry = ""
        try:
            info = tk.info or {}
            sector = str(info.get("sector") or "")
            sub_industry = str(info.get("industry") or "")
        except Exception:
            pass

        roa = None
        roe = None
        if profit is not None and assets not in (None, 0):
            roa = profit / assets
        if profit is not None and equity not in (None, 0):
            roe = profit / equity

        obs = FinancialObservation(
            ticker=symbol.replace(".JK", ""),
            sector=sector,
            sub_industry=sub_industry,
            sub_industry_code="",
            fs_date=period,
            fiscal_year_end="",
            statement_type="quarterly" if quarterly else "annual",
            auditor_opinion="",
            assets=assets,
            liabilities=liabilities,
            equity=equity,
            sales=revenue,
            ebt=ebt,
            profit=profit,
            profit_attributed=None,
            eps=eps,
            book_value=None,
            pe=None,
            pbv=None,
            debt_equity=None,
            roa=roa,
            roe=roe,
            npm=(profit / revenue) if profit is not None and revenue not in (None, 0) else None,
        )

        payload = obs.to_dict()
        payload.update({
            "provider": self.name,
            "source_url": f"{self.source_url}/quote/{symbol}/financials/",
            "retrieved_at_utc": datetime.utcnow().isoformat() + "Z",
            "publication_date": None,
            "event_time_eligible": False,
        })
        (self.cache_dir / f"{symbol}_{'quarterly' if quarterly else 'annual'}.json").write_text(
            __import__("json").dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        return obs
