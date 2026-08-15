from __future__ import annotations
import math
from statistics import mean, pstdev
from dataclasses import dataclass

@dataclass(frozen=True)
class ScoreRow:
    ticker: str
    expected_return: float
    probability_gain: float
    confidence: float
    downside_risk: float
    volatility: float
    momentum: float
    sentiment: float
    event_sample: int
    price_sample: int
    data_quality: float
    score: float
    label: str
    reasons: tuple[str, ...]

def _returns(closes: list[float]) -> list[float]:
    return [b/a-1.0 for a,b in zip(closes, closes[1:]) if a and math.isfinite(a) and math.isfinite(b)]

def _max_drawdown(closes: list[float]) -> float:
    peak = None
    worst = 0.0
    for x in closes:
        if peak is None or x > peak: peak = x
        if peak:
            worst = min(worst, x/peak - 1.0)
    return abs(worst)

def score_ticker(ticker: str, prices: list[dict], news: list[dict], horizon: int, min_price: int, min_events: int) -> ScoreRow:
    closes = [float(x["close"]) for x in prices if x.get("close") is not None]
    rs = _returns(closes)
    price_n = len(rs)
    if price_n < min_price:
        raise ValueError(f"{ticker}: insufficient price history ({price_n})")

    momentum = closes[-1] / closes[max(0, len(closes)-21)] - 1.0 if len(closes) >= 21 else 0.0
    vol = pstdev(rs) * math.sqrt(252) if len(rs) > 1 else 0.0
    downside = pstdev([r for r in rs if r < 0]) * math.sqrt(252) if sum(r < 0 for r in rs) > 1 else 0.0
    drawdown = _max_drawdown(closes)

    valid_news = [n for n in news if n.get("sentiment") is not None]
    sentiment = mean([float(n["sentiment"]) for n in valid_news]) if valid_news else 0.0

    # Transparent baseline: historical forward-return proxy from recent momentum,
    # modulated by current sentiment. It is deliberately bounded and labeled
    # BASELINE until a validated ML artifact is supplied.
    expected = 0.60 * momentum + 0.40 * sentiment * min(0.08, max(0.01, vol * 0.50))
    expected = max(-0.25, min(0.25, expected))

    probability = 0.50 + 0.35 * math.tanh(expected / 0.05)
    sample_factor = min(1.0, math.log1p(price_n + len(valid_news)) / math.log1p(300))
    data_quality = min(1.0, 0.5 + 0.5 * sample_factor)
    confidence = min(1.0, 0.35 + 0.50 * sample_factor + 0.15 * min(1.0, len(valid_news)/10))
    downside_risk = min(1.0, downside + drawdown)

    score = max(0.0, expected) * probability * confidence * data_quality / (1.0 + downside_risk)
    eligible = len(valid_news) >= min_events and price_n >= min_price and data_quality >= 0.5
    if not eligible:
        label = "INSUFFICIENT_EVIDENCE"
    elif score >= 0.03:
        label = "STRONG_OPPORTUNITY"
    elif score >= 0.01:
        label = "MODERATE_OPPORTUNITY"
    else:
        label = "WATCH"

    reasons = []
    if sentiment > 0.2: reasons.append("recent sentiment positive")
    elif sentiment < -0.2: reasons.append("recent sentiment negative")
    if momentum > 0.03: reasons.append("20D momentum positive")
    elif momentum < -0.03: reasons.append("20D momentum negative")
    if downside_risk > 0.15: reasons.append("downside risk elevated")
    if len(valid_news) < min_events: reasons.append("limited news evidence")
    reasons.append("prediction source: transparent baseline")
    return ScoreRow(ticker, expected, probability, confidence, downside_risk, vol,
                    momentum, sentiment, len(valid_news), price_n, data_quality,
                    score, label, tuple(reasons))
