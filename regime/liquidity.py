"""Liquidity headroom distribution for production vs the candidate.

confirm_trade_entry already enforces
    candle quote volume >= max(1000, 10 x trade quote size)
so every trade in both arms cleared 10x. The question promotion raises is
whether the CANDIDATE's extra 4.5x trades sit close to that boundary, or in
thinner pairs, or bunched in time.
"""
import sys, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "user_data/strategies")
BT = Path("user_data/backtest_results"); DATA = Path("user_data/data/binanceus")

ARMS = {"PRODUCTION": ("NNNC_MLX", {95:"W1",181:"W2",56:"W3"}),
        "CANDIDATE":  ("NNNC_MLX_Open", {431:"W1",763:"W2",295:"W3"})}

def load(strat, n):
    for p in sorted(BT.glob("*.zip"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            z=zipfile.ZipFile(p)
            m=[x for x in z.namelist() if x.endswith(".json") and "config" not in x and "market" not in x]
            d=json.loads(z.read(m[0]))
            if strat in d.get("strategy",{}) and len(d["strategy"][strat]["trades"])==n:
                return d["strategy"][strat]["trades"]
        except Exception: continue
    return None

cache={}
def bars(pair):
    if pair not in cache:
        df=pd.read_feather(DATA/f"{pair.replace('/','_')}-15m.feather")
        df["date"]=pd.to_datetime(df["date"],utc=True)
        cache[pair]=df.set_index("date")
    return cache[pair]

out={}
for label,(strat,windows) in ARMS.items():
    rows=[]
    for n,w in windows.items():
        tr=load(strat,n)
        if tr is None: print(f"  {label} {w}: archive not found"); continue
        for t in tr:
            try: df=bars(t["pair"])
            except Exception: continue
            ts=pd.to_datetime(t["open_date"],utc=True)
            if ts not in df.index:
                i=df.index.searchsorted(ts)
                if i>=len(df): continue
                ts=df.index[i]
            r=df.loc[ts]
            if isinstance(r,pd.DataFrame): r=r.iloc[0]
            qv=float(r["volume"])*float(r["close"])
            size=t["stake_amount"]
            rows.append({"pair":t["pair"],"window":w,"qv":qv,"size":size,
                         "headroom":qv/size if size else np.nan,
                         "profit":t["profit_abs"],"date":ts})
    out[label]=pd.DataFrame(rows)

print("HEADROOM = candle quote volume / trade quote size   (filter requires >= 10)\n")
print(f"{'arm':<13}{'n':>6}{'p1':>9}{'p5':>9}{'p10':>9}{'p25':>9}{'median':>10}{'<15x':>8}{'<20x':>8}")
for k,d in out.items():
    h=d["headroom"].dropna()
    q=lambda p: np.percentile(h,p)
    print(f"{k:<13}{len(h):>6}{q(1):>9.1f}{q(5):>9.1f}{q(10):>9.1f}{q(25):>9.1f}{q(50):>10.1f}"
          f"{100*(h<15).mean():>7.1f}%{100*(h<20).mean():>7.1f}%")

print("\nP&L held in the THINNEST trades (would be lost at a stricter filter)")
print(f"{'arm':<13}{'total P&L':>11}{'P&L <15x':>11}{'share':>8}{'P&L <20x':>11}{'share':>8}")
for k,d in out.items():
    tot=d["profit"].sum()
    for_lt=lambda x: d[d["headroom"]<x]["profit"].sum()
    print(f"{k:<13}{tot:>11.1f}{for_lt(15):>11.1f}{100*for_lt(15)/tot:>7.1f}%"
          f"{for_lt(20):>11.1f}{100*for_lt(20)/tot:>7.1f}%")

print("\nCONCURRENCY — simultaneous open positions is capped at 10 by config")
for k,d in out.items():
    dd=d.sort_values("date")
    same_bar = dd.groupby("date").size()
    print(f"  {k:<12} max entries in one 15m bar: {same_bar.max()}   "
          f"bars with >=3 entries: {(same_bar>=3).sum()}")
