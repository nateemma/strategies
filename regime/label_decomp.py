"""How much of the gbb label is TRIGGER (observable now) vs OUTCOME (unknowable)?

labels_gbb = (guard_metric < -0.2) & (bb_width > 0.035) & (future_gain >= thr)

The first two conjuncts are deterministic functions of features the model
already receives at time t. Only the third requires prediction. If the model has
learned the trigger and nothing about the outcome, its precision should equal
the base rate P(future_gain >= thr | trigger).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "user_data/strategies")
DATA = Path("user_data/data/binanceus")
from utils.DataframePopulator import DataframePopulator, DatasetType

H, THR = 48, 0.02
GUARD_T, BBW_T = -0.2, 0.035

rows = []
for pair in ("XRP_USDT", "SOL_USDT", "LINK_USDT"):
    df = pd.read_feather(DATA / f"{pair}-15m.feather")
    df = DataframePopulator().add_indicators(df, dataset_type=DatasetType.MINIMAL)
    close = np.asarray(df["close"], dtype=float)
    gm = np.nan_to_num(np.asarray(df.get("guard_metric", 0.0), dtype=float))
    bw = np.nan_to_num(np.asarray(df.get("bb_width", 0.0), dtype=float))

    fmax = pd.Series(close[::-1]).rolling(H, min_periods=1).max()[::-1].to_numpy()
    fgain = (fmax - close) / close

    trigger = (gm < GUARD_T) & (bw > BBW_T)
    outcome = fgain >= THR
    label = trigger & outcome
    valid = ~np.isnan(fgain)

    n = valid.sum()
    rows.append(dict(pair=pair, n=n,
                     trig=100*trigger[valid].mean(),
                     lab=100*label[valid].mean(),
                     p_out_given_trig=100*outcome[valid & trigger].mean(),
                     p_out_base=100*outcome[valid].mean()))

print("gbb label decomposition, H=48, threshold 0.02\n")
print(f"{'pair':<12}{'bars':>9}{'trigger %':>11}{'label %':>10}"
      f"{'P(gain|trigger)':>17}{'P(gain) all bars':>18}")
for r in rows:
    print(f"{r['pair']:<12}{r['n']:>9}{r['trig']:>10.2f}%{r['lab']:>9.2f}%"
          f"{r['p_out_given_trig']:>16.1f}%{r['p_out_base']:>17.1f}%")
m = lambda k: np.mean([r[k] for r in rows])
print(f"\n  mean P(gain >= {THR:.0%} | trigger) = {m('p_out_given_trig'):.1f}%")
print(f"  mean P(gain) on ALL bars      = {m('p_out_base'):.1f}%")
print(f"\n  the trigger itself lifts the hit rate from {m('p_out_base'):.1f}% to "
      f"{m('p_out_given_trig'):.1f}%")
print(f"  MODEL val_precision = 45.0%")
