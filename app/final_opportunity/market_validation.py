"""Validation and coverage checks for local IDX EOD data."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

REQUIRED = ("date", "open", "high", "low", "close", "volume")


def parse_date(value: str) -> date:
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt).date()
        except ValueError:
            pass
    raise ValueError(f"unsupported date: {value}")


def number(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if s in ("", "-", "--", "null", "None"):
        return None
    return float(s)


@dataclass
class ValidationResult:
    ticker: str
    status: str
    rows: int
    first_date: Optional[str]
    last_date: Optional[str]
    duplicates: int
    invalid_dates: int
    missing_ohlcv: int
    invalid_ohlc: int
    invalid_volume: int
    future_dates: int
    unsorted_dates: int
    message: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def validate_file(path: str | Path, as_of: Optional[date] = None) -> ValidationResult:
    path = Path(path)
    ticker = path.stem.upper()
    as_of = as_of or date.today()

    if not path.exists():
        return ValidationResult(
            ticker, "invalid", 0, None, None, 0, 0, 0, 0, 0, 0, 0, "file not found"
        )

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = {f.strip() for f in (reader.fieldnames or []) if f}
            missing = [c for c in REQUIRED if c not in fields]
            if missing:
                return ValidationResult(
                    ticker, "invalid", 0, None, None, 0, 0, 0, 0, 0, 0, 0,
                    "missing columns: " + ", ".join(missing)
                )

            dates: List[date] = []
            invalid_dates = missing_ohlcv = invalid_ohlc = invalid_volume = 0
            future_dates = unsorted = duplicates = 0
            seen = set()
            previous = None

            for raw in reader:
                try:
                    d = parse_date(raw["date"])
                except Exception:
                    invalid_dates += 1
                    continue

                if d in seen:
                    duplicates += 1
                seen.add(d)

                if previous and d < previous:
                    unsorted += 1
                previous = d

                if d > as_of:
                    future_dates += 1

                vals = {k: number(raw.get(k)) for k in ("open", "high", "low", "close", "volume")}
                if any(vals[k] is None for k in ("open", "high", "low", "close", "volume")):
                    missing_ohlcv += 1
                else:
                    o, h, l, c, v = (vals[k] for k in ("open", "high", "low", "close", "volume"))
                    if not (h >= l and h >= o and h >= c and l <= o and l <= c and c > 0):
                        invalid_ohlc += 1
                    if v < 0:
                        invalid_volume += 1

                dates.append(d)

        dates.sort()
        first = dates[0].isoformat() if dates else None
        last = dates[-1].isoformat() if dates else None
        bad = any((invalid_dates, missing_ohlcv, invalid_ohlc, invalid_volume, future_dates, duplicates))
        return ValidationResult(
            ticker=ticker,
            status="invalid" if bad else "valid",
            rows=len(dates),
            first_date=first,
            last_date=last,
            duplicates=duplicates,
            invalid_dates=invalid_dates,
            missing_ohlcv=missing_ohlcv,
            invalid_ohlc=invalid_ohlc,
            invalid_volume=invalid_volume,
            future_dates=future_dates,
            unsorted_dates=unsorted,
            message="OK" if not bad else "validation errors found",
        )
    except Exception as exc:
        return ValidationResult(
            ticker, "invalid", 0, None, None, 0, 0, 0, 0, 0, 0, 0, str(exc)
        )


def validate_directory(data_dir: str = "data/market_eod", as_of: Optional[date] = None) -> Dict:
    paths = sorted(Path(data_dir).glob("*.csv"))
    results = [validate_file(p, as_of=as_of) for p in paths]
    valid = sum(r.status == "valid" for r in results)
    return {
        "data_dir": str(data_dir),
        "files": len(results),
        "valid": valid,
        "invalid": len(results) - valid,
        "coverage_pct": round(valid / len(results) * 100, 2) if results else 0.0,
        "results": [r.to_dict() for r in results],
    }
