from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

EVENT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class NewsEvent:
    """Canonical ticker-level event used by downstream market analysis."""

    event_id: str
    news_id: str
    ticker: str
    company: Optional[str]
    title: str
    summary: str
    url: str
    source: str
    published_at: Optional[str]
    published_at_utc: Optional[str]
    published_timezone: str
    sentiment: Optional[str]
    sentiment_score: Optional[float]
    signed_score: float
    entity_confidence: Optional[float]
    matched_alias: Optional[str]
    schema_version: str = EVENT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "schema_version": self.schema_version,
            "news_id": self.news_id,
            "ticker": self.ticker,
            "company": self.company,
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at,
            "published_at_utc": self.published_at_utc,
            "published_timezone": self.published_timezone,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "signed_score": self.signed_score,
            "entity_confidence": self.entity_confidence,
            "matched_alias": self.matched_alias,
        }


def validate_event(event: Dict[str, Any]) -> None:
    required = (
        "event_id",
        "news_id",
        "ticker",
        "title",
        "url",
        "source",
        "published_timezone",
        "schema_version",
    )
    missing = [key for key in required if not event.get(key)]
    if missing:
        raise ValueError(f"Missing required event fields: {', '.join(missing)}")

    if not isinstance(event["ticker"], str):
        raise ValueError("ticker must be a string")

    if event.get("sentiment_score") is not None:
        score = float(event["sentiment_score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError("sentiment_score must be between 0 and 1")

    if event.get("entity_confidence") is not None:
        confidence = float(event["entity_confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("entity_confidence must be between 0 and 1")
