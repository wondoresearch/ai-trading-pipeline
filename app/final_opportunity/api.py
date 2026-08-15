from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from .config import Config
from .service import ResearchService

CONFIG = Config()
app = FastAPI(title="Opportunity Research", version="1.0")
service = ResearchService(CONFIG)

@app.get("/", response_class=HTMLResponse)
def index():
    path = Path(__file__).resolve().parents[2] / "web" / "index.html"
    return path.read_text(encoding="utf-8")

@app.get("/api/universe")
def universe():
    return service.store.universe()

@app.get("/api/news/{ticker}")
def news(ticker: str):
    return service.store.news(ticker.upper(), 50)

@app.post("/api/sync")
def sync(tickers: list[str]):
    return service.sync(tickers)

@app.get("/api/ranking")
def ranking(
    tickers: str = Query(..., description="Comma-separated tickers"),
    horizon: int = 20,
    refresh: bool = False,
):
    selected = [x.strip().upper() for x in tickers.split(",") if x.strip()]
    if refresh:
        service.sync(selected)
    return service.analyze(selected, horizon)
