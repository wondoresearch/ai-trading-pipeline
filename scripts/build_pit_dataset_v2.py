
#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.final_opportunity.backtest.pit_v2 import audit_pit_states
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("-o","--output",required=True)
    args=ap.parse_args()
    rows=json.loads(Path(args.input).read_text())
    audit=audit_pit_states(rows)
    if audit["status"]!="PASS":
        raise SystemExit("PIT AUDIT FAILED: "+json.dumps(audit))
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(rows,indent=2))
    print(json.dumps({"rows":len(rows),"output":str(out),"audit":audit},indent=2))
if __name__=="__main__": main()
