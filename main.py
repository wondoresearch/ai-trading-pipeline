import os
import sys
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

from app.config import TICKER_MASTER_FILE, OUTPUT_DIR, get_rss_feeds
from app.pipeline import NewsPipeline
from app.storage import save_json


def main():
    print("=== AI Trading News Pipeline ===")
    print("Loading model and collecting RSS news...")

    pipeline = NewsPipeline(TICKER_MASTER_FILE)
    results = pipeline.run(get_rss_feeds())

    output = OUTPUT_DIR / "news_result.json"
    save_json(results, output)

    print(f"Matched articles: {len(results)}")
    print(f"Saved: {output}")

    for article in results[:10]:
        print("-" * 80)
        print(article["title"])
        print("Ticker:", [x["ticker"] for x in article["ticker_mapping"]])
        print(
            "Sentiment:",
            article["sentiment"]["label"],
            f"({article['sentiment']['score']:.3f})"
        )

if __name__ == "__main__":
    main()
