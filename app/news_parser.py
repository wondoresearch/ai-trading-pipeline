from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Dict, List

from app.event_schema import NewsEvent, validate_event
from app.time_utils import normalize_timezone, normalize_utc


def load_news(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_event_id(news_id: str, ticker: str) -> str:
    value = f"{news_id}|{ticker}".encode("utf-8")
    return sha256(value).hexdigest()


def extract_news_events(news) -> List[Dict[str, Any]]:
    """Convert article-level results into canonical ticker-level events."""
    events = []

    for article in news:
        news_id = str(article.get("id", "")).strip()
        if not news_id:
            continue

        published_at_utc = normalize_utc(article.get("published_at"))
        published_at_local = (
            normalize_timezone(published_at_utc, "Asia/Jakarta")
            if published_at_utc
            else None
        )

        sentiment = article.get("sentiment") or {}
        mappings = article.get("ticker_mapping") or []

        for mapping in mappings:
            ticker = str(mapping.get("ticker", "")).strip().upper()
            if not ticker:
                continue

            sentiment_score = sentiment.get("score")
            if sentiment_score is not None:
                sentiment_score = float(sentiment_score)

            event = NewsEvent(
                event_id=generate_event_id(news_id, ticker),
                news_id=news_id,
                ticker=ticker,
                company=mapping.get("company"),
                title=article.get("title", ""),
                summary=article.get("summary", ""),
                url=article.get("url", ""),
                source=article.get("source", "RSS"),
                published_at=published_at_local,
                published_at_utc=published_at_utc,
                published_timezone="Asia/Jakarta",
                sentiment=sentiment.get("label"),
                sentiment_score=sentiment_score,
                signed_score=float(sentiment.get("signed_score", 0.0) or 0.0),
                entity_confidence=mapping.get("entity_confidence"),
                matched_alias=mapping.get("matched_alias"),
            ).to_dict()

            validate_event(event)
            events.append(event)

    return events
