from __future__ import annotations
import hashlib, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

POSITIVE = {
    "naik","tumbuh","meningkat","melonjak","laba","untung","positif","buyback",
    "dividen","ekspansi","pulih","menguat","surplus","rebound","growth","profit",
    "strong","beat","upgrade","bullish"
}
NEGATIVE = {
    "turun","merosot","rugi","negatif","anjlok","utang","gagal","krisis","fraud",
    "sanksi","melemah","defisit","downgrade","bearish","loss","weak","miss"
}

def lexicon_sentiment(text: str) -> tuple[float, str]:
    words = {x.strip(".,:;!?()[]{}").lower() for x in text.split()}
    p, n = len(words & POSITIVE), len(words & NEGATIVE)
    if p == n:
        return 0.0, "neutral"
    score = (p - n) / max(1, p + n)
    return max(-1.0, min(1.0, score)), ("positive" if score > 0 else "negative")

def _parse_date(value: str) -> str:
    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()

def fetch_google_news(ticker: str, company_name: str | None = None, days: int = 30) -> list[dict]:
    q = ticker if not company_name else f'"{ticker}" OR "{company_name}"'
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({
        "q": q, "hl":"id", "gl":"ID", "ceid":"ID:id"
    })
    req = urllib.request.Request(url, headers={"User-Agent":"ai-trading-opportunity-research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        root = ET.fromstring(r.read())
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        pub = _parse_date(item.findtext("pubDate") or "")
        try:
            dt = datetime.fromisoformat(pub)
        except Exception:
            dt = datetime.now(timezone.utc)
        if dt < cutoff:
            continue
        score, label = lexicon_sentiment(title)
        news_id = hashlib.sha256(f"{ticker}|{title}|{link}|{pub}".encode()).hexdigest()
        rows.append({
            "news_id": news_id,
            "ticker": ticker.upper(),
            "title": title,
            "url": link,
            "source": "google_news_rss",
            "published_at": pub,
            "available_at": pub,
            "sentiment": score,
            "sentiment_label": label,
            "sentiment_source": "lexicon_baseline_v1",
        })
    return rows
