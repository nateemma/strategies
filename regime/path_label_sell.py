"""Mirror of path_label.py, for the SELL label.

The buy label promises "close rises >= min_gain within H" and its path risk is
that the ATR STOP fires first, so the promised gain is unreachable. That was
measured in path_label.py: 1.5-4.2% of buy labels are killed that way.

The sell label is the exact mirror and has never been examined, because
`enable_exit_signal` has always been False on NNMT -- the sell head never
reached a trade. The 2026-09-06 exit sweep changed that (exit signal ON +
exit_rvol 1.0 is the one positive result on this family), so the sell label's
path behaviour now matters.

THE MIRROR RISK. The sell label says "close falls >= min_loss within H" and is
used to EXIT a long. Its path risk is that the TAKE-PROFIT fires first: the
position closes profitably on the way, and the sell signal never gets to act.
So where the buy side measures the worst ADVERSE excursion before the target,
this measures the best FAVOURABLE excursion before the target.

A sell label whose MFE-before-target exceeds the take-profit is PRE-EMPTED --
not wrong, just moot. A high pre-emption rate means the sell head is being
trained largely on bars where its opinion could never be expressed.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "user_data/strategies")
DATA = Path("user_data/data/binanceus")
from utils.DataframePopulator import DataframePopulator, DatasetType

H, GUARD_T_SELL, BBW_T = 48, 0.2, 0.035
TARGETS   = [0.007, 0.02]            # MIN_SELL_LOSS_THRESHOLD: pre- and post-inheritance
PREEMPTS  = [0.022, 0.030, 0.032]    # cexit_take_profit now / roi / pre-inheritance
PAIRS = ["XRP","SOL","LINK","AVAX","NEAR","DOT","LTC","BCH","AAVE","SUI","ZEC"]

def mfe_before_target(close, horizon, target):
    """Best FAVOURABLE excursion before the first bar where the DROP >= target.

    Mirror of path_label.mae_before_target. NaN when the drop never happens
    inside the horizon (that bar is not a label anyway).
    """
    n = len(close); out = np.full(n, np.nan)
    for t in range(n):
        end = min(n, t + horizon + 1)
        if t + 1 >= end: continue
        fut = close[t+1:end]; c0 = close[t]
        drops = (c0 - fut) / c0
        hit = np.flatnonzero(drops >= target)
        if hit.size == 0: continue
        upto = fut[:hit[0]+1]
        out[t] = (np.max(upto) - c0) / c0      # best rise on the way down
    return out

rows = []
for base in PAIRS:
    try:
        df = pd.read_feather(DATA / f"{base}_USDT-15m.feather")
    except Exception:
        continue
    df = DataframePopulator().add_indicators(df, dataset_type=DatasetType.MINIMAL)
    close = np.asarray(df["close"], dtype=float)
    gm = np.nan_to_num(np.asarray(df.get("guard_metric", 0.0), dtype=float))
    bw = np.nan_to_num(np.asarray(df.get("bb_width", 0.0), dtype=float))
    fmin  = pd.Series(close[::-1]).rolling(H, min_periods=1).min()[::-1].to_numpy()
    floss = (close - fmin) / close
    for tgt in TARGETS:
        label = (gm > GUARD_T_SELL) & (bw > BBW_T) & (floss >= tgt) & ~np.isnan(floss)
        if label.sum() < 50: continue
        rows.append(pd.DataFrame({"pair": base, "target": tgt,
                                  "mfe": mfe_before_target(close, H, tgt)[label]}))

d = pd.concat(rows, ignore_index=True)
print("SELL LABEL — how often is the drop PRE-EMPTED by the take-profit?\n")
print(f"{'min_loss':<10}{'labelled bars':>15}{'never drops':>13}"
      + "".join(f"{'preempt@'+str(round(p*100,1))+'%':>16}" for p in PREEMPTS))
for tgt in TARGETS:
    v = d.loc[d.target == tgt, "mfe"]; n = len(v); ok = v.notna()
    row = f"{tgt*100:>5.1f}%{'':<4}{n:>15,}{(n-ok.sum())/n*100:>12.1f}%"
    for p in PREEMPTS:
        row += f"{100*(v >= p).sum()/n:>15.1f}%"
    print(row)
print("\n  preempt@X = the long would have hit a +X% take-profit BEFORE the sell")
print("  label's drop arrived, so the sell signal could never have acted.")
print("\nMFE-before-drop distribution (min_loss 0.02):")
v = d.loc[d.target == 0.02, "mfe"].dropna()
for q in (50, 75, 90, 95, 99):
    print(f"    p{q:<3} {np.percentile(v, q)*100:+.2f}%")
