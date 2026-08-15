
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

    def to_dict(self):
        d = asdict(self)
        for k in ("as_of", "publication_date", "financial_period_end"):
            if d.get(k) is not None:
                d[k] = d[k].isoformat()
        return d

def latest_eligible_financial(history: Iterable[FinancialHistory], as_of: date):
    candidates = [
        x for x in history
        if x.ticker
        and x.publication_date is not None
        and x.publication_date <= as_of
        and x.financial_period_end <= as_of
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
):
    fin = latest_eligible_financial(financial_history, as_of)
    # A PIT row may exist without fundamentals; this is not a fabricated score.
    return PITState(
        ticker=ticker.upper().replace(".JK", ""),
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
    )

def audit_pit_states(rows):
    seen = set()
    duplicates = []
    lookahead = []
    unknown_pub = []
    for r in rows:
        key = (r["ticker"], r["as_of"])
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        pub = r.get("publication_date")
        if pub and pub > r["as_of"]:
            lookahead.append(key)
        if r.get("fundamental_score") is not None and not pub:
            unknown_pub.append(key)
    return {
        "status": "PASS" if not duplicates and not lookahead and not unknown_pub else "FAIL",
        "rows": len(rows),
        "eligible": sum(bool(r.get("eligible")) for r in rows),
        "duplicate_observations": len(duplicates),
        "lookahead_violations": len(lookahead),
        "unknown_financial_publication": len(unknown_pub),
        "violations_sample": [str(x) for x in (duplicates + lookahead + unknown_pub)[:10]],
    }
