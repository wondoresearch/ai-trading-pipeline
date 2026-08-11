import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
TICKER_MASTER_FILE = DATA_DIR / "ticker_master.json"

DEFAULT_RSS_FEEDS = [
    "https://news.google.com/rss/search?q=BBCA+saham&hl=id&gl=ID&ceid=ID%3Aid",
    "https://news.google.com/rss/search?q=BBRI+saham&hl=id&gl=ID&ceid=ID%3Aid",
    "https://news.google.com/rss/search?q=BMRI+saham&hl=id&gl=ID&ceid=ID%3Aid",
    "https://news.google.com/rss/search?q=TLKM+saham&hl=id&gl=ID&ceid=ID%3Aid",
    "https://news.google.com/rss/search?q=IHSG+saham&hl=id&gl=ID&ceid=ID%3Aid",
]

def get_rss_feeds():
    raw = os.getenv("RSS_FEEDS", "").strip()
    if not raw:
        return DEFAULT_RSS_FEEDS
    return [x.strip() for x in raw.split(",") if x.strip()]

def get_poll_interval():
    try:
        return max(30, int(os.getenv("POLL_INTERVAL_SECONDS", "300")))
    except ValueError:
        return 300
