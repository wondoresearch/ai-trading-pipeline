
"""Deterministic, JSON-safe opportunity report serialization."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.opportunity_pipeline import OpportunityPipelineResult


def _safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def build_report(result: OpportunityPipelineResult, top_n: int = 10) -> dict[str, Any]:
    ranking = []
    for item in result.ranking[:top_n]:
        ranking.append({
            "rank": item.rank,
            "ticker": item.ticker,
            "expected_return": item.prediction,
            "confidence": item.confidence,
            "risk": {
                "volatility": item.risk.volatility,
                "downside_deviation": item.risk.downside_deviation,
                "max_drawdown": item.risk.max_drawdown,
                "beta": item.risk.beta,
                "observation_count": item.risk.observation_count,
            },
            "opportunity": {
                "risk_penalty": item.opportunity.risk_penalty,
                "score": item.opportunity.score,
            },
        })

    report = {
        "schema_version": "1.0",
        "purpose": "research_opportunity_ranking",
        "live_trading": False,
        "universe": list(result.universe.tickers),
        "data_status": [
            {
                "ticker": item.ticker,
                "eligible": item.eligible,
                "observation_count": item.observation_count,
                "reason": item.reason,
            }
            for item in result.data_status
        ],
        "ranking": ranking,
    }
    return _safe(report)


def write_report(
    result: OpportunityPipelineResult,
    path: Path,
    top_n: int = 10,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(result, top_n=top_n)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        handle.write("\n")
    return path
