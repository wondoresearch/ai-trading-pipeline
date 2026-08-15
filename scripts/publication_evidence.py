#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen
from html import unescape

BASE_URL="https://idx.sahamidx.com/lk/?y={year}"
ENTRY_RE=re.compile(r"(?P<ticker>[A-Z0-9]{2,6})\s*-\s*(?P<period_year>\d{4})\s*-\s*(?P<type>Audit|TW1|TW2|TW3|TW4)",re.I)
TS_RE=re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")

def fetch_html(year):
    req=Request(BASE_URL.format(year=year),headers={"User-Agent":"ai-trading-news-pipeline/1.0 research"})
    with urlopen(req,timeout=30) as r:return r.read().decode("utf-8","replace")

def normalize_html(s):
    return re.sub(r"\s+"," ",re.sub(r"<[^>]+>"," ",unescape(s)))

def parse_publication_evidence(html,tickers=None):
    text=normalize_html(html)
    wanted={x.upper().replace(".JK","") for x in tickers} if tickers else None
    out=[]
    for m in ENTRY_RE.finditer(text):
        ticker=m.group("ticker").upper()
        if wanted and ticker not in wanted: continue
        tail=text[m.end():m.end()+350]
        tm=TS_RE.search(tail)
        if not tm: continue
        ts=datetime.strptime(tm.group(),"%Y-%m-%d %H:%M:%S")
        out.append({
            "ticker":ticker,
            "financial_year":int(m.group("period_year")),
            "statement_type":m.group("type").upper(),
            "publication_date":ts.date().isoformat(),
            "publication_timestamp":ts.isoformat(),
            "source":"idx_sahamidx_public_listing",
            "source_url":BASE_URL,
            "evidence_level":"listing_timestamp"
        })
    return list({(x["ticker"],x["financial_year"],x["statement_type"],x["publication_timestamp"]):x for x in out}.values())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tickers",required=True); ap.add_argument("--year",type=int,required=True)
    ap.add_argument("--fixture"); ap.add_argument("-o","--output")
    a=ap.parse_args()
    html=Path(a.fixture).read_text(encoding="utf-8") if a.fixture else fetch_html(a.year)
    p={"provider":"idx_sahamidx_public_listing","requested_year":a.year,
       "retrieved_at_utc":datetime.now(timezone.utc).isoformat(),
       "rows":parse_publication_evidence(html,a.tickers.split(","))}
    s=json.dumps(p,indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True,exist_ok=True)
        Path(a.output).write_text(s,encoding="utf-8")
    print(s)

if __name__=="__main__": main()
