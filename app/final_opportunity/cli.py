from __future__ import annotations

import argparse
import json

from .config import Config
from .market_import import import_market_file
from .service import ResearchService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="final-opportunity")
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="sync market EOD data and Google News RSS")
    sync.add_argument("--tickers", nargs="+", required=True)

    imp = sub.add_parser("import-market", help="import an IDX EOD CSV/ZIP")
    imp.add_argument("--file", required=True)
    imp.add_argument(
        "--output-dir",
        default="data/market_eod",
        help="normalized per-ticker CSV directory",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "import-market":
        print(json.dumps(
            import_market_file(args.file, args.output_dir),
            indent=2,
            sort_keys=True,
        ))
        return

    service = ResearchService(Config())
    if args.command == "sync":
        print(json.dumps(service.sync(args.tickers), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
