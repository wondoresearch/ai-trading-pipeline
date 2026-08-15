
#!/usr/bin/env python3
import argparse, json
from datetime import date
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.final_opportunity.backtest.pit_v2 import FinancialHistory, build_pit_state, audit_pit_states

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON containing as_of/market and financial history")
    ap.add_argument("-o","--output",required=True)
    args=ap.parse_args()
    src=json.loads(Path(args.input).read_text())
    rows=[]
    for x in src:
        history=[]
        for f in x.get("financial_history", []):
            history.append(FinancialHistory(
                ticker=x["ticker"],
                financial_period_end=date.fromisoformat(f["financial_period_end"]),
                publication_date=date.fromisoformat(f["publication_date"]) if f.get("publication_date") else None,
                fundamental_score=f.get("fundamental_score"),
                source=f.get("source",""),
            ))
        rows.append(build_pit_state(
            x["ticker"], date.fromisoformat(x["as_of"]),
            price=x.get("price"),
            market_score=x.get("market_score"),
            market_source=x.get("market_source"),
            news_score=x.get("news_score"),
            news_source=x.get("news_source"),
            financial_history=history,
        ).to_dict())
    audit=audit_pit_states(rows)
    if audit["status"] != "PASS":
        raise SystemExit("PIT AUDIT FAILED: "+json.dumps(audit))
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(rows,indent=2))
    print(json.dumps({"rows":len(rows),"output":str(out),"audit":audit},indent=2))
if __name__=="__main__": main()
