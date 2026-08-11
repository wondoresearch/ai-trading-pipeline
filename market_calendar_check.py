import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from app.market_calendar import IDXMarketCalendar


def main():
    parser = argparse.ArgumentParser(
        description="Inspect IDX trading calendar context for a timestamp."
    )
    parser.add_argument(
        "timestamp",
        help="ISO timestamp, e.g. 2026-08-11T10:00:00+07:00",
    )
    args = parser.parse_args()

    dt = datetime.fromisoformat(args.timestamp)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("Asia/Jakarta"))

    calendar = IDXMarketCalendar()
    context = calendar.context_at(dt)

    for key, value in context.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
