import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"

if "AI_TRADING_BOOTSTRAPPED" not in os.environ:
    if VENV_PYTHON.exists():
        current_python = Path(sys.executable).resolve()
        if current_python != VENV_PYTHON.resolve():
            try:
                import numpy  # type: ignore
                import transformers  # type: ignore
            except ModuleNotFoundError:
                os.environ["AI_TRADING_BOOTSTRAPPED"] = "1"
                os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

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
