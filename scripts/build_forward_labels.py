#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from app.final_opportunity.backtest.forward_labels import load_price_csv, build_forward_labels, audit_labels

def main():
    ap=argparse.ArgumentParser(description='Build strict point-in-time forward-return labels.')
    ap.add_argument('input'); ap.add_argument('-o','--output',required=True); ap.add_argument('--price-dir',default='data/market_eod'); ap.add_argument('--horizons',default='1,3,5,10,20')
    a=ap.parse_args(); obs=json.loads(Path(a.input).read_text())
    tickers=sorted({str(x['ticker']).upper() for x in obs}); prices={}
    for t in tickers:
        p=Path(a.price_dir)/f'{t}.csv'
        if p.exists(): prices[t]=load_price_csv(p,t)
        else: prices[t]=[]
    rows=build_forward_labels(obs,prices,tuple(int(x) for x in a.horizons.split(',') if x.strip()))
    audit=audit_labels(rows); Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps({'rows':rows,'audit':audit},indent=2))
    print(json.dumps({'output':a.output,'audit':audit},indent=2)); raise SystemExit(0 if audit['status']=='PASS' else 1)
if __name__=='__main__': main()
