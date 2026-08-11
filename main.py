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
