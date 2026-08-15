from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    db_path: Path = Path(os.getenv("OPPORTUNITY_DB", "data/opportunity.db"))
    output_dir: Path = Path(os.getenv("OPPORTUNITY_OUTPUT", "output"))
    timezone: str = "Asia/Jakarta"
    history_days: int = int(os.getenv("OPPORTUNITY_HISTORY_DAYS", "1825"))
    news_days: int = int(os.getenv("OPPORTUNITY_NEWS_DAYS", "30"))
    min_price_observations: int = int(os.getenv("OPPORTUNITY_MIN_PRICE_OBS", "120"))
    min_event_observations: int = int(os.getenv("OPPORTUNITY_MIN_EVENT_OBS", "5"))
    stale_hours: int = int(os.getenv("OPPORTUNITY_STALE_HOURS", "36"))
    top_n: int = int(os.getenv("OPPORTUNITY_TOP_N", "10"))

    def validate(self) -> None:
        if self.history_days < 60:
            raise ValueError("history_days must be >= 60")
        if self.min_price_observations < 20:
            raise ValueError("min_price_observations must be >= 20")
