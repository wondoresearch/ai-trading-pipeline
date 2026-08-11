from __future__ import annotations

from json import JSONDecodeError
import logging

import pandas as pd
import requests
import yfinance as yf
from requests.exceptions import RequestException
from yfinance.utils import UnknownTimeZoneError

logger = logging.getLogger(__name__)


class PriceProvider:
    """Yahoo-backed daily price provider with a stable provider boundary."""

    def __init__(self, market_suffix=".JK"):
        self.market_suffix = market_suffix
        self.cache = {}

    def _yf_symbol(self, ticker: str) -> str:
        ticker = str(ticker).strip().upper()
        if not ticker:
            raise ValueError("ticker must not be empty")
        return ticker if ticker.endswith(self.market_suffix) else f"{ticker}{self.market_suffix}"

    def _fetch_yahoo_chart(self, yf_ticker, start, end):
        params = {
            "period1": int(pd.Timestamp(start).timestamp()),
            "period2": int(pd.Timestamp(end).timestamp()),
            "interval": "1d",
            "events": "div,split",
        }
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{yf_ticker}"
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        chart = data.get("chart", {})
        if chart.get("error"):
            return pd.DataFrame()

        result = chart.get("result")
        if not result:
            return pd.DataFrame()

        result = result[0]
        timestamps = result.get("timestamp")
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        adjclose = (
            result.get("indicators", {})
            .get("adjclose", [{}])[0]
            .get("adjclose")
        )

        if not timestamps or not quote or adjclose is None:
            return pd.DataFrame()

        return pd.DataFrame(
            {
                "Date": pd.to_datetime(timestamps, unit="s"),
                "Open": quote.get("open"),
                "High": quote.get("high"),
                "Low": quote.get("low"),
                "Close": quote.get("close"),
                "Adj Close": adjclose,
                "Volume": quote.get("volume"),
            }
        )

    @staticmethod
    def _normalize_dataframe(df):
        if df.empty:
            return df

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        date_column = "Date" if "Date" in df.columns else "Datetime"

        if date_column not in df.columns:
            raise ValueError("Price data does not contain Date/Datetime column")

        parsed = pd.to_datetime(df[date_column], errors="coerce")
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
        else:
            parsed = parsed.dt.tz_localize(None)

        df["Date"] = parsed
        return (
            df.dropna(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )

    def get_history(self, ticker, start, end):
        yf_ticker = self._yf_symbol(ticker)
        cache_key = (yf_ticker, str(start), str(end))

        if cache_key in self.cache:
            return self.cache[cache_key]

        df = pd.DataFrame()

        try:
            df = self._fetch_yahoo_chart(yf_ticker, start, end)
        except (
            JSONDecodeError,
            UnknownTimeZoneError,
            ValueError,
            RequestException,
        ) as exc:
            logger.warning("Yahoo chart fetch failed for %s: %s", yf_ticker, exc)

        if df.empty:
            try:
                df = yf.download(
                    yf_ticker,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    timeout=10,
                )
            except (
                JSONDecodeError,
                UnknownTimeZoneError,
                ValueError,
                ConnectionError,
                TimeoutError,
                RequestException,
            ) as exc:
                logger.warning("yfinance download failed for %s: %s", yf_ticker, exc)

        if df.empty:
            try:
                df = yf.Ticker(yf_ticker).history(
                    start=start,
                    end=end,
                    auto_adjust=False,
                    actions=False,
                    interval="1d",
                )
            except (
                JSONDecodeError,
                UnknownTimeZoneError,
                ValueError,
                ConnectionError,
                TimeoutError,
                RequestException,
            ) as exc:
                logger.warning(
                    "yfinance history fallback failed for %s: %s",
                    yf_ticker,
                    exc,
                )
                return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        try:
            df = self._normalize_dataframe(df)
        except ValueError as exc:
            logger.warning("Invalid price data for %s: %s", yf_ticker, exc)
            return pd.DataFrame()

        self.cache[cache_key] = df
        return df
