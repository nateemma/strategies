"""Is the MFE-only label teaching the model to buy bars that would have stopped out?

For each labelled buy bar, walk forward to the bar where the maximum favourable
excursion occurs, and measure the WORST drawdown along the way. If that exceeds
the operative stop, the labelled profit was never reachable in practice.

The operative stop for NN strategies is the ATR band, NOT self.stoploss --
NNNC clamps it to [-6%, -4%], so -4% is the optimistic bound and -6% the loose one.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "user_data/strategies")
from utils.DataframePopulator import DataframePopulator, DatasetType
from Framework.TrainingSignals import get_train_buy_signals, LabelMethod

DATA = Path("user_data/data/binanceus")
H = 48

def mae_before_mfe(close: np.ndarray, idx: np.ndarray, horizon: int):
    """For each index, drawdown from entry to the running MFE point."""
    worst = np.full(len(idx), np.nan)
    for k, t in enumerate(idx):
        end = min(len(close), t + horizon + 1)
        fut = close[t + 1:end]
        if len(fut) == 0:
            continue
        j = int(np.nanargmax(fut))               # bar where MFE occurs
        path = fut[: j + 1]                      # everything up to and including it
        worst[k] = (np.nanmin(path) - close[t]) / close[t] * 100
    return worst

rows = []
for pair in ("XRP_USDT", "SOL_USDT", "LINK_USDT"):
    df = pd.read_feather(DATA / f"{pair}-15m.feather")
    df = DataframePopulator().add_indicators(df, dataset_type=DatasetType.MINIMAL)
    close = np.asarray(df["close"], dtype=float)
    for name, method in (("gbb", LabelMethod.gbb), ("breakout_gbb", LabelMethod.breakout_gbb)):
        y = get_train_buy_signals(df, method=method,
                                  params={"horizon": H, "min_gain": 0.005, "min_loss": 0.005})
        idx = np.where(np.asarray(y) > 0.5)[0]
        if len(idx) == 0:
            continue
        w = mae_before_mfe(close, idx, H)
        w = w[~np.isnan(w)]
        rows.append(dict(pair=pair.split("_")[0], label=name, n=len(w),
                         median=np.median(w), p25=np.percentile(w, 25),
                         p10=np.percentile(w, 10),
                         below4=(w <= -4).mean() * 100, below6=(w <= -6).mean() * 100))
r = pd.DataFrame(rows)
print("Drawdown BEFORE the labelled profit is reachable (%), by label\n")
print(r.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print("\npooled:")
g = r.groupby("label").apply(
    lambda s: pd.Series({
        "n": s.n.sum(),
        "median_dd": np.average(s["median"], weights=s.n),
        "p10_dd": np.average(s.p10, weights=s.n),
        "% labels needing >4% drawdown": np.average(s.below4, weights=s.n),
        "% labels needing >6% drawdown": np.average(s.below6, weights=s.n),
    }), include_groups=False)
print(g.to_string(float_format=lambda v: f"{v:.2f}"))
