"""Local fundamental-history cache and feature preparation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .fundamental_engine import build_fundamental_features


class FundamentalHistory:
    def __init__(self, cache_dir: str = "data/financial_history"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, ticker: str, rows: list[Any]) -> Path:
        path = self.cache_dir / f"{ticker.upper().replace('.JK','')}.json"
        payload = []
        for row in rows:
            if hasattr(row, "to_dict"):
                payload.append(row.to_dict())
            elif isinstance(row, dict):
                payload.append(row)
            else:
                payload.append(dict(row.__dict__))
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def load_dicts(self, ticker: str) -> list[dict]:
        path = self.cache_dir / f"{ticker.upper().replace('.JK','')}.json"
        if not path.exists():
            return []
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []

    @staticmethod
    def _sort_key(row: Any) -> str:
        if isinstance(row, dict):
            return str(row.get("fiscal_year_end") or row.get("fs_date") or "")
        return str(getattr(row, "fiscal_year_end", "") or getattr(row, "fs_date", ""))

    @classmethod
    def feature_from_rows(cls, rows: list[Any]):
        if not rows:
            return None
        ordered = sorted(rows, key=cls._sort_key, reverse=True)
        return build_fundamental_features(
            ordered[0], ordered[1] if len(ordered) > 1 else None
        )
