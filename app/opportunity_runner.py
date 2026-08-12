
"""Research-only opportunity ranking runner.

This runner intentionally accepts prepared, point-in-time prediction and
historical-return data. It does not invent market/news data and does not
place trades.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.opportunity_config import OpportunityRunnerConfig
from app.opportunity_pipeline import OpportunityPipeline
from app.opportunity_report import write_report


def load_tickers(path: Path) -> list[str]:
    values = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            values.append(value)
    return values


def load_prepared_data(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(
    tickers: Sequence[str],
    prepared_data: dict,
    config: OpportunityRunnerConfig | None = None,
):
    config = config or OpportunityRunnerConfig()

    pipeline = OpportunityPipeline()
    result = pipeline.run(
        tickers=tickers,
        predictions=prepared_data["predictions"],
        confidence=prepared_data["confidence"],
        historical_returns=prepared_data["historical_returns"],
        market_returns=prepared_data.get("market_returns"),
        data_minimum_observations=config.minimum_history,
        risk_minimum_observations=config.risk_minimum_history,
    )
    write_report(result, config.output_path, top_n=config.top_n)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run research-only risk-adjusted stock opportunity ranking."
    )
    parser.add_argument(
        "--universe",
        required=True,
        type=Path,
        help="Text file containing one ticker per line.",
    )
    parser.add_argument(
        "--prepared-data",
        required=True,
        type=Path,
        help="JSON containing predictions, confidence and historical_returns.",
    )
    parser.add_argument(
        "--output",
        default="output/opportunity_ranking.json",
        type=Path,
    )
    parser.add_argument("--top-n", default=10, type=int)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    tickers = load_tickers(args.universe)
    prepared = load_prepared_data(args.prepared_data)

    config = OpportunityRunnerConfig(
        output_path=args.output,
        top_n=args.top_n,
    )
    result = run(tickers, prepared, config=config)

    print(f"Universe: {len(result.universe)}")
    print(f"Eligible: {len(result.ranking)}")
    print(f"Output: {config.output_path}")

    for item in result.ranking[:config.top_n]:
        print(
            f"{item.rank:>3}  {item.ticker:<8} "
            f"expected={item.prediction:.4f} "
            f"confidence={item.confidence:.2f} "
            f"score={item.opportunity.score:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
