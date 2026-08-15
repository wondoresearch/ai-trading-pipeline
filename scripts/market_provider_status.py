#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from app.final_opportunity.market import FreeIDXMarketData
from app.final_opportunity.provider_status import check_provider

def main():
    p=argparse.ArgumentParser(description="Show market provider coverage and freshness.")
    p.add_argument("--data-dir",default="data/market_eod")
    p.add_argument("--tickers",nargs="+",default=["BBRI","BBCA","BMRI","BBNI","TLKM"])
    p.add_argument("--max-age-days",type=int,default=3)
    p.add_argument("--json",action="store_true")
    a=p.parse_args()
    r=check_provider(FreeIDXMarketData(a.data_dir),a.tickers,a.max_age_days).as_dict()
    if a.json:
        print(json.dumps(r,indent=2)); return 0 if r["status"]=="healthy" else 1
    print("MARKET PROVIDER STATUS"); print("-"*72)
    for label,key in [("Provider","provider"),("Available","available"),("Status","status"),("Fresh","fresh"),
                      ("Tickers","tickers"),("Requested","requested_tickers"),("Rows","rows"),
                      ("Oldest date","oldest_date"),("Latest date","latest_date"),("Age days","age_days"),
                      ("Max age","max_age_days"),("Missing","missing_tickers"),("Stale","stale_tickers"),("Message","message")]:
        print(f"{label:14}: {r.get(key)}")
    print("-"*72); return 0 if r["status"]=="healthy" else 1
if __name__=="__main__": raise SystemExit(main())
