from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf


class MarketReturnProvider:
    """Yahoo-backed IDX Composite (IHSG) close-price provider."""

    SYMBOL = "^JKSE"
    PRICE_FIELD = "Close"

    def __init__(self) -> None:
        self.cache = {}
        self.last_error: Optional[dict] = None

    @staticmethod
    def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        if isinstance(frame.columns, pd.MultiIndex):
            close_column = next(
                (column for column in frame.columns
                 if "Close" in {str(part) for part in column}),
                None,
            )
            if close_column is None:
                raise ValueError("Market data must contain Close")
            frame = frame[[close_column]].copy()
            frame.columns = ["Close"]
        frame = frame.reset_index()
        date_column = "Date" if "Date" in frame.columns else "Datetime"
        if date_column not in frame or "Close" not in frame:
            raise ValueError("Market data must contain Date/Datetime and Close")
        frame["Date"] = pd.to_datetime(frame[date_column], errors="coerce")
        if getattr(frame["Date"].dt, "tz", None) is not None:
            frame["Date"] = frame["Date"].dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
        return frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    def get_history(self, start, end) -> pd.DataFrame:
        key = (str(start), str(end))
        if key in self.cache:
            return self.cache[key]
        self.last_error = None
        try:
            frame = yf.download(self.SYMBOL, start=start, end=end, auto_adjust=False,
                                progress=False, threads=False, timeout=10)
            frame = self._normalize(frame)
        except Exception as exc:
            self.last_error = {"stage": "market_download", "error_type": type(exc).__name__, "message": str(exc)}
            return pd.DataFrame()
        self.cache[key] = frame
        return frame

    def get_history_with_status(self, start, end):
        frame = self.get_history(start, end)
        if self.last_error is not None:
            return frame, "provider_error"
        return frame, "ok" if not frame.empty else "empty"
