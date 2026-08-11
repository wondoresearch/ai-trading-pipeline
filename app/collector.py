from datetime import datetime, timezone
from hashlib import sha256
import feedparser

def generate_id(title: str, url: str) -> str:
    value = f"{title.strip()}|{url.strip()}".encode("utf-8")
    return sha256(value).hexdigest()

def collect_news(feed_urls):
    articles = []
    for feed_url in feed_urls:
        feed = feedparser.parse(feed_url)
        source = feed.feed.get("title", "RSS")
        for item in feed.entries:
            title = item.get("title", "").strip()
            url = item.get("link", "").strip()
            if not title or not url:
                continue
            published = item.get("published") or item.get("updated")
            articles.append({
                "id": generate_id(title, url),
                "source": source,
                "title": title,
                "summary": item.get("summary", "").strip(),
                "url": url,
                "published_at": published,
                "collected_at": datetime.now(timezone.utc).isoformat()
            })
    return articles
