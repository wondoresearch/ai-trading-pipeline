from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Optional, Iterable


@dataclass(frozen=True)
class FinancialHistory:
    ticker: str
    financial_period_end: date
    publication_date: Optional[date]
    fundamental_score: Optional[float]
    source: str = ""
    evidence_level: Optional[str] = None
    publication_timestamp: Optional[str] = None
    source_url: Optional[str] = None


@dataclass(frozen=True)
class PITState:
    ticker: str
    as_of: date
    publication_date: Optional[date]
    financial_period_end: Optional[date]
    price: Optional[float]
    market_score: Optional[float]
    news_score: Optional[float]
    fundamental_score: Optional[float]
    eligible: bool
    financial_source: Optional[str] = None
    market_source: Optional[str] = None
    news_source: Optional[str] = None
    financial_evidence_level: Optional[str] = None
    financial_publication_timestamp: Optional[str] = None
    financial_source_url: Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        for k in ("as_of", "publication_date", "financial_period_end"):
            if d.get(k) is not None:
                d[k] = d[k].isoformat()
        return d


def _normalize_ticker(value: str) -> str:
    return str(value or "").upper().replace(".JK", "")


def latest_eligible_financial(
    ticker: str,
    history: Iterable[FinancialHistory],
    as_of: date,
    *,
    require_evidence: bool = False,
):
    """Return the latest financial fact for ``ticker`` provably available by ``as_of``."""
    normalized_ticker = _normalize_ticker(ticker)
    candidates = [
        x for x in history
        if _normalize_ticker(x.ticker) == normalized_ticker
        and x.publication_date is not None
        and x.publication_date <= as_of
        and x.financial_period_end <= as_of
        and (not require_evidence or bool(x.evidence_level))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda x: (x.financial_period_end, x.publication_date))


def build_pit_state(
    ticker: str,
    as_of: date,
    *,
    price: Optional[float] = None,
    market_score: Optional[float] = None,
    market_source: Optional[str] = None,
    news_score: Optional[float] = None,
    news_source: Optional[str] = None,
    financial_history: Iterable[FinancialHistory] = (),
    require_financial_evidence: bool = False,
):
    fin = latest_eligible_financial(
        ticker,
        financial_history,
        as_of,
        require_evidence=require_financial_evidence,
    )
    return PITState(
        ticker=_normalize_ticker(ticker),
        as_of=as_of,
        publication_date=fin.publication_date if fin else None,
        financial_period_end=fin.financial_period_end if fin else None,
        price=price,
        market_score=market_score,
        news_score=news_score,
        fundamental_score=fin.fundamental_score if fin else None,
        eligible=True,
        financial_source=fin.source if fin else None,
        market_source=market_source,
        news_source=news_source,
        financial_evidence_level=fin.evidence_level if fin else None,
        financial_publication_timestamp=fin.publication_timestamp if fin else None,
        financial_source_url=fin.source_url if fin else None,
    )


def audit_pit_states(rows):
    """Audit a reconstructed snapshot under the strict historical PIT contract."""
    seen = set()
    duplicates = []
    lookahead = []
    unknown_pub = []
    unevidenced_financial = []

    for r in rows:
        key = (_normalize_ticker(r.get("ticker")), r["as_of"])
        if key in seen:
            duplicates.append(key)
        seen.add(key)

        pub = r.get("publication_date")
        if pub and pub > r["as_of"]:
            lookahead.append(key)

        if r.get("fundamental_score") is not None:
            if not pub:
                unknown_pub.append(key)
            if not r.get("financial_evidence_level"):
                unevidenced_financial.append(key)

    violations = duplicates + lookahead + unknown_pub + unevidenced_financial
    return {
        "status": "PASS" if not violations else "FAIL",
        "rows": len(rows),
        "eligible": sum(bool(r.get("eligible")) for r in rows),
        "duplicate_observations": len(duplicates),
        "lookahead_violations": len(lookahead),
        "unknown_financial_publication": len(unknown_pub),
        "unevidenced_financial_observations": len(unevidenced_financial),
        "violations_sample": [str(x) for x in violations[:10]],
    }
