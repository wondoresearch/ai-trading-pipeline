from __future__ import annotations
from datetime import datetime, timedelta, timezone
import json, math
from .config import Config
from .market import YahooFinanceMarketData
from .news import fetch_google_news
from .scoring import score_ticker
from .store import Store

class ResearchService:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.store = Store(self.config.db_path)
        self.market = YahooFinanceMarketData()

    @staticmethod
    def _clean_json(value):
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return value
        if isinstance(value, dict):
            return {k: ResearchService._clean_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ResearchService._clean_json(v) for v in value]
        return value

    def sync(self, tickers: list[str]) -> dict:
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=self.config.history_days)).date().isoformat()
        end = (now + timedelta(days=1)).date().isoformat()
        clean = sorted({t.strip().upper().replace(".JK", "") for t in tickers if t.strip()})
        if not clean:
            raise ValueError("At least one ticker is required")
        selected = self.market.stocks(clean)
        self.store.upsert_universe(selected)
        price_count = news_count = 0
        errors = []
        for item in selected:
            ticker = item["ticker"]
            try:
                prices = self.market.history(ticker, start, end)
                self.store.upsert_prices(prices)
                price_count += len(prices)
            except Exception as exc:
                errors.append({"ticker": ticker, "stage": "market", "error": str(exc)})
            try:
                news = fetch_google_news(ticker, item.get("name"), self.config.news_days)
                self.store.upsert_news(news)
                news_count += len(news)
            except Exception as exc:
                errors.append({"ticker": ticker, "stage": "news", "error": str(exc)})
        return {"tickers":len(selected), "prices":price_count, "news":news_count, "errors":errors,
                "source":"yahoo_finance + google_news_rss"}

    def analyze(self, tickers: list[str], horizon: int | None = None, as_of: str | None = None) -> dict:
        horizon = horizon or 20
        cutoff = as_of or datetime.now(timezone.utc).isoformat()
        rows = []
        errors = []
        for ticker in sorted({x.strip().upper().replace(".JK", "") for x in tickers if x.strip()}):
            try:
                row = score_ticker(
                    ticker, self.store.prices(ticker), self.store.news(ticker, 100, cutoff),
                    horizon, self.config.min_price_observations, self.config.min_event_observations
                )
                rows.append(row)
            except Exception as exc:
                errors.append({"ticker":ticker, "error":str(exc)})
        rows.sort(key=lambda x: (-x.score, x.ticker))
        result = {
            "schema_version":"final-1.1",
            "purpose":"research_opportunity_ranking",
            "live_trading":False,
            "data_source":"yahoo_finance_personal_research",
            "as_of":cutoff,
            "horizon_days":horizon,
            "ranking":[{
                "rank":i+1, "ticker":r.ticker, "score":r.score,
                "expected_return":r.expected_return, "probability_gain":r.probability_gain,
                "confidence":r.confidence, "downside_risk":r.downside_risk,
                "volatility":r.volatility, "momentum_20d":r.momentum,
                "sentiment":r.sentiment, "event_sample":r.event_sample,
                "price_sample":r.price_sample, "data_quality":r.data_quality,
                "label":r.label, "reasons":list(r.reasons),
            } for i,r in enumerate(rows)],
            "errors":errors,
        }
        result = self._clean_json(result)
        self.store.save_ranking(cutoff, horizon, result["ranking"])
        return result
