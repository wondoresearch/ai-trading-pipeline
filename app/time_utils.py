from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from dateutil import parser
from dateutil.tz import tzutc

DEFAULT_TIMEZONE = "Asia/Jakarta"


def parse_datetime(value: object) -> Optional[datetime]:
    """Parse a timestamp into a timezone-aware datetime.

    Naive timestamps are treated as UTC because a feed timestamp without an
    offset cannot safely be assumed to be Jakarta time.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            dt = parser.parse(text)
        except (TypeError, ValueError, OverflowError):
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tzutc())

    return dt


def normalize_utc(value: object) -> Optional[str]:
    dt = parse_datetime(value)
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def normalize_timezone(
    value: object,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> Optional[str]:
    dt = parse_datetime(value)
    if dt is None:
        return None

    from zoneinfo import ZoneInfo

    try:
        target = ZoneInfo(timezone_name)
    except Exception as exc:
        raise ValueError(f"Invalid timezone: {timezone_name}") from exc

    return dt.astimezone(target).isoformat()
