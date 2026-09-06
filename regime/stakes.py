"""Capital deployment under the volume cap: production vs candidate.

custom_stake_amount CAPS the stake at 1/10 of the candle's quote volume rather
than rejecting the trade, so a "liquidity problem" here shows up as SHRUNKEN
STAKES, not as missing trades. Configured stake is 900 on a 10000 wallet.
"""
import json, zipfile
from pathlib import Path
import numpy as np, pandas as pd
BT=Path("user_data/backtest_results")
ARMS={"PRODUCTION":("NNNC_MLX",{95:"W1",181:"W2",56:"W3"}),
      "CANDIDATE":("NNNC_MLX_Open",{431:"W1",763:"W2",295:"W3"})}
def load(strat,n):
    for p in sorted(BT.glob("*.zip"),key=lambda q:q.stat().st_mtime,reverse=True):
        try:
            z=zipfile.ZipFile(p); m=[x for x in z.namelist() if x.endswith(".json") and "config" not in x and "market" not in x]
            d=json.loads(z.read(m[0]))
            if strat in d.get("strategy",{}) and len(d["strategy"][strat]["trades"])==n:
                return d["strategy"][strat]["trades"]
        except Exception: continue
FULL=900.0
res={}
for lab,(strat,ws) in ARMS.items():
    rows=[]
    for n,w in ws.items():
        tr=load(strat,n)
        if not tr: continue
        for t in tr: rows.append({"w":w,"stake":t["stake_amount"],"profit":t["profit_abs"],
                                  "ratio":t["profit_ratio"],"pair":t["pair"]})
    res[lab]=pd.DataFrame(rows)

print("STAKE SIZE — configured 900 on a 10,000 wallet\n")
print(f"{'arm':<13}{'n':>6}{'median':>9}{'mean':>9}{'p10':>8}{'p90':>9}{'at full':>9}{'<10% full':>11}")
for k,d in res.items():
    s=d["stake"]
    print(f"{k:<13}{len(s):>6}{s.median():>9.1f}{s.mean():>9.1f}{np.percentile(s,10):>8.1f}"
          f"{np.percentile(s,90):>9.1f}{100*(s>=FULL*0.99).mean():>8.1f}%{100*(s<FULL*0.1).mean():>10.1f}%")

print("\nCAPITAL DEPLOYED and return ON DEPLOYED capital")
print(f"{'arm':<13}{'total staked':>14}{'total P&L':>11}{'ret on staked':>15}{'wallet ret%':>13}")
for k,d in res.items():
    tot=d["stake"].sum(); pnl=d["profit"].sum()
    wallet = 24.13 if k=="PRODUCTION" else 34.10
    print(f"{k:<13}{tot:>14,.0f}{pnl:>11.1f}{100*pnl/tot:>14.2f}%{wallet:>12.2f}%")

print("\nWHERE THE P&L COMES FROM, by stake bucket")
for k,d in res.items():
    print(f"  {k}")
    d=d.copy(); d["b"]=pd.cut(d["stake"],[0,50,150,450,899,1e9],
                              labels=["<50","50-150","150-450","450-900","full 900"])
    g=d.groupby("b",observed=True).agg(n=("profit","size"),pnl=("profit","sum"),med=("stake","median"))
    tot=d["profit"].sum()
    for b,r in g.iterrows():
        print(f"    {str(b):<10}{int(r['n']):>6} trades  median stake {r['med']:>7.1f}"
              f"  P&L {r['pnl']:>9.1f}  ({100*r['pnl']/tot:>5.1f}%)")
