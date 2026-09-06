"""How many gbb labels are unreachable because the stop fires FIRST?

The label says "close rises >= 2% within 48 bars". It says nothing about the
path, so a bar can reach +2% having first dropped through the ATR stop
(-4%..-6%). The oracle exposed exactly this: 47 of 237 perfectly-predicted
trades exited at -5.65%.

The right test is NOT whole-horizon MAE (a drop AFTER the take-profit fires is
irrelevant -- we already exited). It is the worst adverse excursion BEFORE the
target is first reached.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "user_data/strategies")
DATA = Path("user_data/data/binanceus")
from utils.DataframePopulator import DataframePopulator, DatasetType

H, TGT, GUARD_T, BBW_T = 48, 0.02, -0.2, 0.035
EXITS = [0.020, 0.024, 0.032]
STOPS = [-0.04, -0.05, -0.06]
PAIRS = ["XRP","SOL","LINK","AVAX","NEAR","DOT","LTC","BCH","AAVE","SUI","ZEC"]

def mae_before_target(close, horizon, target):
    """Worst adverse excursion before the first bar where gain >= target."""
    n=len(close); out=np.full(n, np.nan)
    for t in range(n):
        end=min(n, t+horizon+1)
        if t+1>=end: continue
        fut=close[t+1:end]; c0=close[t]
        gains=(fut-c0)/c0
        hit=np.flatnonzero(gains>=target)
        if hit.size==0: continue          # target never reached -> not a label anyway
        upto=fut[:hit[0]+1]
        out[t]=(np.min(upto)-c0)/c0       # most negative excursion on the way
    return out

rows=[]
for base in PAIRS:
    try:
        df=pd.read_feather(DATA/f"{base}_USDT-15m.feather")
    except Exception: continue
    df=DataframePopulator().add_indicators(df, dataset_type=DatasetType.MINIMAL)
    close=np.asarray(df["close"],dtype=float)
    gm=np.nan_to_num(np.asarray(df.get("guard_metric",0.0),dtype=float))
    bw=np.nan_to_num(np.asarray(df.get("bb_width",0.0),dtype=float))
    fmax=pd.Series(close[::-1]).rolling(H,min_periods=1).max()[::-1].to_numpy()
    fgain=(fmax-close)/close
    label=(gm<GUARD_T)&(bw>BBW_T)&(fgain>=TGT)&~np.isnan(fgain)
    if label.sum()<50: continue
    rec={"pair":base}
    for e in EXITS:
        rec[f"mae_{int(e*1000)}"]=mae_before_target(close,H,e)[label]
    rows.append(pd.DataFrame(rec))

d=pd.concat(rows,ignore_index=True)
print("LABEL TARGET vs the level the EXIT actually needs\n")
print(f"{'must reach':<14}{'reachable':>12}{'never reached':>15}"
      + "".join(f"{'killed@'+str(int(s*100))+'%':>14}" for s in STOPS))
for e in EXITS:
    c=f"mae_{int(e*1000)}"; v=d[c]
    ok=v.notna(); n=len(v)
    row=f"{e*100:>6.1f}%{'':<7}{ok.sum():>12,}{(n-ok.sum())/n*100:>14.1f}%"
    for s in STOPS:
        row+=f"{100*(v<=s).sum()/n:>13.1f}%"
    print(row)
print("\n  'never reached' = the trade never hits that level within the horizon at all,")
print("  so it runs to a stop, an unclog or max_hold instead.")
print("  the oracle stopped out on 47/237 = 19.8% of its trades.")
