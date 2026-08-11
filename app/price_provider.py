import json
from json import JSONDecodeError

import pandas as pd
import requests
import yfinance as yf
from requests.exceptions import RequestException
from yfinance.utils import UnknownTimeZoneError


class PriceProvider:

    def __init__(self):
        self.cache = {}

    def _fetch_yahoo_chart(self, yf_ticker, start, end):
        params = {
            "period1": int(pd.Timestamp(start).timestamp()),
            "period2": int(pd.Timestamp(end).timestamp()),
            "interval": "1d",
            "events": "div,split",
        }
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{yf_ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}

        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        chart = data.get("chart", {})
        if chart.get("error"):
            return pd.DataFrame()

        result = chart.get("result")
        if not result:
            return pd.DataFrame()

        result = result[0]
        timestamps = result.get("timestamp")
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        adjclose = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")

        if not timestamps or not quote or adjclose is None:
            return pd.DataFrame()

        df = pd.DataFrame({
            "Date": pd.to_datetime(timestamps, unit="s"),
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Adj Close": adjclose,
            "Volume": quote.get("volume"),
        })

        return df

    def get_history(
        self,
        ticker,
        start,
        end
    ):

        yf_ticker = f"{ticker}.JK"

        cache_key = (
            yf_ticker,
            str(start),
            str(end)
        )

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
            Exception,
        ) as exc:
            print(
                f"Failed yahoo chart fetch for '{yf_ticker}' reason: {exc}"
            )

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
                Exception,
            ) as exc:
                print(
                    f"Failed yfinance download for '{yf_ticker}' reason: {exc}"
                )

        if df.empty:
            try:
                ticker_obj = yf.Ticker(yf_ticker)
                df = ticker_obj.history(
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
                Exception,
            ) as exc:
                print(
                    f"Failed fallback download for '{yf_ticker}' reason: {exc}"
                )
                return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # yfinance versi tertentu menghasilkan MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        df["Date"] = pd.to_datetime(
            df["Date"]
        ).dt.tz_localize(None)

        self.cache[cache_key] = df

        return df