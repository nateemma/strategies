"""The ceiling-vs-learnability trade-off along the guard threshold.

Ceiling comes from the oracle gate (measured, regime/LABEL_TRIGGER_GATE.md).
The "free MCC" -- what a trigger-only predictor scores with no forecasting at
all -- is computable in closed form from label densities, because
`label = trigger AND outcome` makes label a strict subset of trigger, so recall
is 100% and the confusion matrix follows from two counts.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "user_data/strategies")
DATA = Path("user_data/data/binanceus")
from utils.DataframePopulator import DataframePopulator, DatasetType

H, THR, BBW_T = 48, 0.02, 0.035
GUARDS = [-0.4, -0.2, 0.0, 0.2, 0.5, 1.0]
CEIL_W1 = {-0.2: 26.54, 0.0: 35.09, 0.2: 43.60, 1.0: 97.97}

acc = {g: {"trig": 0, "lab": 0, "n": 0} for g in GUARDS}
for pair in ("XRP_USDT", "SOL_USDT", "LINK_USDT"):
    df = pd.read_feather(DATA / f"{pair}-15m.feather")
    df = DataframePopulator().add_indicators(df, dataset_type=DatasetType.MINIMAL)
    close = np.asarray(df["close"], dtype=float)
    gm = np.nan_to_num(np.asarray(df.get("guard_metric", 0.0), dtype=float))
    bw = np.nan_to_num(np.asarray(df.get("bb_width", 0.0), dtype=float))
    fmax = pd.Series(close[::-1]).rolling(H, min_periods=1).max()[::-1].to_numpy()
    fgain = (fmax - close) / close
    valid = ~np.isnan(fgain)
    outcome = fgain >= THR
    for g in GUARDS:
        trig = (gm < g) & (bw > BBW_T)
        acc[g]["trig"] += int((trig & valid).sum())
        acc[g]["lab"]  += int((trig & outcome & valid).sum())
        acc[g]["n"]    += int(valid.sum())

print("Trigger relaxation: what you GAIN in ceiling vs what you LOSE for free\n")
print(f"{'guard <':>9}{'trigger %':>11}{'label %':>10}{'P(gain|trig)':>14}"
      f"{'free MCC':>10}{'W1 ceiling':>12}")
for g in GUARDS:
    a = acc[g]; n = a["n"]
    TP = a["lab"]/n; FP = (a["trig"]-a["lab"])/n; TN = 1-TP-FP
    mcc = (TP*TN) / np.sqrt((TP+FP)*TP*(TN+FP)*TN) if TP>0 and TN>0 and (TP+FP)>0 else float("nan")
    c = CEIL_W1.get(g)
    print(f"{g:>9.1f}{100*a['trig']/n:>10.2f}%{100*a['lab']/n:>9.2f}%"
          f"{100*a['lab']/max(a['trig'],1):>13.1f}%{mcc:>10.3f}"
          + (f"{c:>11.2f}%" if c is not None else f"{'--':>12}"))
print("\n  free MCC = a trigger-only predictor, zero forecasting, 100% recall")
print("  W1 ceiling = perfect prediction of that label, guards off (lookahead)")
