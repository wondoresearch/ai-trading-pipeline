"""Phase 13 — deterministic trading signal conversion."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any
import json


class Signal(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NO_POSITION = "NO_POSITION"


@dataclass(frozen=True)
class TradingSignal:
    event_id: str
    ticker: str
    signal: Signal
    probability_positive: float | None
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "signal": self.signal.value,
            "probability_positive": self.probability_positive,
            "threshold": self.threshold,
        }


class TradingSignalEngine:
    """Converts frozen model probabilities into signals; no tuning occurs here."""

    def generate(self, event_id: str, ticker: str, probability_positive: float | None,
                 threshold: float) -> TradingSignal:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        if probability_positive is None:
            signal = Signal.NO_POSITION
        elif not 0 <= probability_positive <= 1:
            raise ValueError("probability_positive must be between 0 and 1")
        elif probability_positive >= threshold:
            signal = Signal.LONG
        elif probability_positive <= 1 - threshold:
            signal = Signal.SHORT
        else:
            signal = Signal.NO_POSITION
        return TradingSignal(event_id, ticker, signal, probability_positive, threshold)
