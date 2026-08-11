import time
from datetime import datetime, timezone

from app.config import (
    TICKER_MASTER_FILE,
    OUTPUT_DIR,
    get_poll_interval,
    get_rss_feeds
)
from app.pipeline import NewsPipeline
from app.storage import append_jsonl

def main():
    pipeline = NewsPipeline(TICKER_MASTER_FILE)
    seen_ids = set()

    print("=== AI Trading News Worker ===")
    print(f"Polling every {get_poll_interval()} seconds")

    while True:
        try:
            results = pipeline.run(get_rss_feeds())
            new_results = []

            for item in results:
                if item["id"] not in seen_ids:
                    seen_ids.add(item["id"])
                    item["processed_at"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    new_results.append(item)

            if new_results:
                append_jsonl(
                    new_results,
                    OUTPUT_DIR / "news_stream.jsonl"
                )
                print(
                    f"[{datetime.now().isoformat()}] "
                    f"New matched news: {len(new_results)}"
                )
            else:
                print(
                    f"[{datetime.now().isoformat()}] No new matched news"
                )

        except Exception as exc:
            print(f"Worker error: {type(exc).__name__}: {exc}")

        time.sleep(get_poll_interval())

if __name__ == "__main__":
    main()
