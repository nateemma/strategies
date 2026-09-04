"""If we set the gain threshold to target the TAIL, what survives?

Constraints that matter:
  - label density must stay above ~5.5% of bars (aug_risk all-Hold collapse floor)
  - the ATR stop is 4-6%, so a labelled move must clear it to be worth taking
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,"user_data/strategies")
from utils.DataframePopulator import DataframePopulator, DatasetType

DATA=Path("user_data/data/binanceus")
PAIRS=("XRP_USDT","SOL_USDT","LINK_USDT")
THR=(0.005,0.010,0.020,0.030,0.050,0.080)
STOP=5.0

def fwd_max(close,H):
    n=len(close); out=np.full(n,np.nan)
    for t in range(n-H-1): out[t]=close[t+1:t+H+1].max()
    return out

for H in (24,48,96):
    tot_bars=0; per={t:[] for t in THR}
    for pair in PAIRS:
        df=pd.read_feather(DATA/f"{pair}-15m.feather")
        df=DataframePopulator().add_indicators(df,dataset_type=DatasetType.MINIMAL)
        close=np.asarray(df.close,dtype=float)
        gm=np.nan_to_num(np.asarray(df.guard_metric,dtype=float))
        bw=np.nan_to_num(np.asarray(df.bb_width,dtype=float))
        cond=(gm>0.2)&(bw>0.035)
        g=(fwd_max(close,H)-close)/close*100
        ok=cond&~np.isnan(g)
        tot_bars+=len(df)
        for t in THR: per[t].append(g[ok&(g>=t*100)])
    print(f"\n=== H={H} ({H*0.25:.0f}h) — labelled population by gain threshold ===")
    print(f"{'thr':>7}{'labels':>9}{'% bars':>8}{'median':>9}{'mean':>8}{'p90':>8}"
          f"{'>stop%':>8}{'reward:risk':>12}")
    for t in THR:
        v=np.concatenate(per[t])
        if len(v)==0: continue
        med=np.median(v); above=(v>=STOP).mean()*100
        print(f"{t:>7}{len(v):>9}{len(v)/tot_bars*100:>7.2f}%{med:>9.2f}{v.mean():>8.2f}"
              f"{np.percentile(v,90):>8.2f}{above:>7.1f}%{med/STOP:>12.2f}")
    print(f"  (aug_risk collapse floor ~5.5% of bars; stop = {STOP}%)")
