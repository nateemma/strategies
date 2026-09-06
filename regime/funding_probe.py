"""Does funding separate gbb outcomes? Cheap feasibility gate before any build.

Constitution VII: gate an expensive build behind a minimal measurement. Adding
funding as a feature means a scaler regen + retrains, and W3 has ZERO funding
coverage. So first: among TRIGGER bars, does the funding rate predict whether
the 2% forward gain arrives?
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "user_data/strategies")
OHLCV = Path("user_data/data/binanceus")
FUND = Path("user_data/data/funding")
from utils.DataframePopulator import DataframePopulator, DatasetType

H, THR, GUARD_T, BBW_T = 48, 0.02, -0.2, 0.035
PAIRS = ["XRP","SOL","LINK","AVAX","NEAR","DOT","LTC","BCH","AAVE","SUI","ZEC"]

rows = []
for base in PAIRS:
    try:
        df = pd.read_feather(OHLCV / f"{base}_USDT-15m.feather")
        fd = pd.read_feather(FUND / f"{base}_funding.feather")
    except Exception:
        continue
    df = DataframePopulator().add_indicators(df, dataset_type=DatasetType.MINIMAL)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    fd["dt"] = pd.to_datetime(fd["dt"], utc=True)
    df = pd.merge_asof(df.sort_values("date"), fd.sort_values("dt").rename(
        columns={"dt":"date"}), on="date", direction="backward")

    close = np.asarray(df["close"], dtype=float)
    gm = np.nan_to_num(np.asarray(df.get("guard_metric",0.0), dtype=float))
    bw = np.nan_to_num(np.asarray(df.get("bb_width",0.0), dtype=float))
    fmax = pd.Series(close[::-1]).rolling(H, min_periods=1).max()[::-1].to_numpy()
    fgain = (fmax-close)/close
    trig = (gm < GUARD_T) & (bw > BBW_T)
    ok = trig & ~np.isnan(fgain) & df["funding"].notna().to_numpy()
    if ok.sum() < 200: continue
    rows.append(pd.DataFrame({"pair":base, "funding":df["funding"].to_numpy()[ok],
                              "gain":fgain[ok], "hit":(fgain>=THR)[ok]}))

d = pd.concat(rows, ignore_index=True)
print(f"trigger bars with funding data: {len(d)} across {d['pair'].nunique()} pairs\n")
d["q"] = pd.qcut(d["funding"], 5, labels=False, duplicates="drop")
print("P(gain >= 2%) among TRIGGER bars, by funding quintile")
print(f"{'quintile':<10}{'n':>8}{'funding med':>14}{'P(gain)':>10}{'mean gain':>12}")
for q,s in d.groupby("q"):
    print(f"{int(q)+1:<10}{len(s):>8}{s['funding'].median():>14.6f}"
          f"{100*s['hit'].mean():>9.1f}%{100*s['gain'].mean():>11.2f}%")
lo,hi = d[d["q"]==d["q"].min()], d[d["q"]==d["q"].max()]
print(f"\n  spread Q1 -> Q5:  P(gain) {100*lo['hit'].mean():.1f}% -> {100*hi['hit'].mean():.1f}%"
      f"   ({100*(hi['hit'].mean()-lo['hit'].mean()):+.1f}pp)")
print(f"  rank correlation funding vs gain: {d['funding'].corr(d['gain'], method='spearman'):+.4f}")
print(f"  rank correlation funding vs hit : {d['funding'].corr(d['hit'].astype(float), method='spearman'):+.4f}")
print("\n  for scale: the guard trigger itself moves P(gain) 34.9% -> 51.1% (+16.2pp)")
