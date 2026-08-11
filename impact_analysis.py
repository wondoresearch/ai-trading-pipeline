import argparse
from pathlib import Path

from app.config import OUTPUT_DIR
from app.impact_pipeline import ImpactPipeline
from app.storage import save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run impact analysis on saved news results."
    )
    parser.add_argument(
        "news_file",
        nargs="?",
        default=OUTPUT_DIR / "news_result.json",
        help="Path to a JSON file containing news results.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR / "impact_result.json",
        help="Path to save the impact analysis results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    news_file = Path(args.news_file)

    if not news_file.exists():
        raise FileNotFoundError(f"News file not found: {news_file}")

    print("=== AI Trading Impact Analysis ===")
    print(f"Loading news from: {news_file}")

    pipeline = ImpactPipeline()
    results = pipeline.run(news_file)

    output_path = Path(args.output)
    save_json(results, output_path)

    print(f"Impact results: {len(results)} events")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
