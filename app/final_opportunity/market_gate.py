"""Backward-compatible market readiness helpers."""
from __future__ import annotations
from typing import Dict
from .market_readiness import MarketReadinessError

MarketDataNotReady = MarketReadinessError

def require_market_ready(status: Dict) -> Dict:
    if (
        status.get("status") != "healthy"
        or not status.get("available")
        or not status.get("fresh")
    ):
        raise MarketDataNotReady(
            type("ReadinessResult", (), {
                "reason": status.get("message", "Market data is not ready."),
                "provider": status.get("provider", "unknown"),
                "status": status.get("status", "unavailable"),
            })()
        )
    return status
