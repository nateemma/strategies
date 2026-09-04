"""Same tail analysis, applied to gbb (method 17) -- the PRODUCTION label."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0,"user_data/strategies")
from utils.DataframePopulator import DataframePopulator, DatasetType
DATA=Path("user_data/data/binanceus"); PAIRS=("XRP_USDT","SOL_USDT","LINK_USDT")

def load(pair):
    df=pd.read_feather(DATA/f"{pair}-15m.feather")
    return DataframePopulator().add_indicators(df,dataset_type=DatasetType.MINIMAL)

def fwdmax(c,H):
    out=np.full(len(c),np.nan)
    for t in range(len(c)-H-1): out[t]=c[t+1:t+H+1].max()
    return out

CACHE={p:load(p) for p in PAIRS}
print("=== gbb (guard_metric < -0.2 & bb_width > 0.035) forward MFE ===\n")
print(f"{'H':>4}{'median':>9}{'p90':>8}{'p99':>8}{'skew':>8}{'exkurt':>9}{'top10%share':>13}")
for H in (6,12,24,48,96):
    allg=[]
    for p in PAIRS:
        df=CACHE[p]; c=np.asarray(df.close,dtype=float)
        gm=np.nan_to_num(np.asarray(df.guard_metric,dtype=float))
        bw=np.nan_to_num(np.asarray(df.bb_width,dtype=float))
        cond=(gm<-0.2)&(bw>0.035)
        g=(fwdmax(c,H)-c)/c*100
        allg.append(g[cond&~np.isnan(g)])
    g=np.concatenate(allg); t10=np.percentile(g,90); tot=g[g>0].sum()
    print(f"{H:>4}{np.median(g):>9.2f}{t10:>8.2f}{np.percentile(g,99):>8.2f}"
          f"{stats.skew(g):>8.2f}{stats.kurtosis(g):>9.1f}{g[g>=t10].sum()/tot*100:>12.1f}%")

print("\n=== labelled population by threshold, vs NNNC's actual exits ===")
print("   NNNC: ATR stop [-6%,-4%], cexit_take_profit 2.4%, ROI 5.0%, unclog at 12h")
print(f"\n{'H':>4}{'thr':>7}{'%bars':>8}{'median':>9}{'clears2.4%':>12}{'clears5%':>10}{'medPeak(h)':>12}{'stopped@-4%':>13}")
for H,thr in ((24,0.007),(48,0.007),(48,0.010),(48,0.020),(96,0.007),(96,0.020)):
    lab=[]; dd=[]; bars=[]; nb=0
    for p in PAIRS:
        df=CACHE[p]; c=np.asarray(df.close,dtype=float); nb+=len(df)
        gm=np.nan_to_num(np.asarray(df.guard_metric,dtype=float))
        bw=np.nan_to_num(np.asarray(df.bb_width,dtype=float))
        cond=(gm<-0.2)&(bw>0.035)
        g=(fwdmax(c,H)-c)/c*100
        idx=np.where(cond&~np.isnan(g)&(g>=thr*100))[0]
        lab.append(g[idx])
        for t in idx:
            fut=c[t+1:min(len(c),t+H+1)]
            if len(fut)==0: continue
            j=int(np.nanargmax(fut)); path=fut[:j+1]
            dd.append((np.nanmin(path)-c[t])/c[t]*100); bars.append(j+1)
    v=np.concatenate(lab); dd=np.array(dd); bars=np.array(bars)
    print(f"{H:>4}{thr:>7}{len(v)/nb*100:>7.2f}%{np.median(v):>9.2f}"
          f"{(v>=2.4).mean()*100:>11.1f}%{(v>=5).mean()*100:>9.1f}%"
          f"{np.median(bars)*0.25:>12.1f}{(dd<=-4).mean()*100:>12.1f}%")
