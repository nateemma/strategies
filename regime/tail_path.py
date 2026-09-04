"""Are the tail-targeted labels REACHABLE, or do they stop out before the peak?

Walks forward to the MFE bar and measures the worst drawdown on the way there.
Operative stop is the ATR band [-6%, -4%], so -4% is the pessimistic bound.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,"user_data/strategies")
from utils.DataframePopulator import DataframePopulator, DatasetType
DATA=Path("user_data/data/binanceus")

def audit(close, idx, H):
    dd=[]; bars=[]
    for t in idx:
        end=min(len(close), t+H+1)
        fut=close[t+1:end]
        if len(fut)==0: continue
        j=int(np.nanargmax(fut))
        path=fut[:j+1]
        dd.append((np.nanmin(path)-close[t])/close[t]*100)
        bars.append(j+1)
    return np.array(dd), np.array(bars)

CASES=[(48,0.005,"CURRENT"),(48,0.02,""),(96,0.03,"CANDIDATE"),(96,0.05,"")]
print(f"{'setting':<22}{'labels':>8}{'medDD':>8}{'p10DD':>8}{'stopped@-4%':>13}{'medBars':>9}{'medHrs':>8}")
for H,thr,tag in CASES:
    ALLdd=[]; ALLb=[]; n=0
    for pair in ("XRP_USDT","SOL_USDT","LINK_USDT"):
        df=pd.read_feather(DATA/f"{pair}-15m.feather")
        df=DataframePopulator().add_indicators(df,dataset_type=DatasetType.MINIMAL)
        close=np.asarray(df.close,dtype=float)
        gm=np.nan_to_num(np.asarray(df.guard_metric,dtype=float))
        bw=np.nan_to_num(np.asarray(df.bb_width,dtype=float))
        cond=(gm>0.2)&(bw>0.035)
        fm=np.full(len(close),np.nan)
        for t in range(len(close)-H-1): fm[t]=close[t+1:t+H+1].max()
        g=(fm-close)/close*100
        idx=np.where(cond & ~np.isnan(g) & (g>=thr*100))[0]
        n+=len(idx)
        d,b=audit(close,idx,H); ALLdd.append(d); ALLb.append(b)
    d=np.concatenate(ALLdd); b=np.concatenate(ALLb)
    lab=f"H={H} thr={thr} {tag}".strip()
    print(f"{lab:<22}{n:>8}{np.median(d):>8.2f}{np.percentile(d,10):>8.2f}"
          f"{(d<=-4).mean()*100:>12.1f}%{np.median(b):>9.0f}{np.median(b)*0.25:>8.1f}")
