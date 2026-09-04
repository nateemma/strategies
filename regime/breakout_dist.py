"""Distribution of forward gains from breakout-condition bars.

Question: is the breakout payoff fat-tailed, and if so does labelling on an
'average' move (min_gain 0.005) target the wrong part of the distribution?
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, "user_data/strategies")
from utils.DataframePopulator import DataframePopulator, DatasetType

DATA = Path("user_data/data/binanceus")
PAIRS = ("XRP_USDT", "SOL_USDT", "LINK_USDT")
HORIZONS = (6, 12, 24, 48, 96)

def fwd_max(close, H):
    n = len(close)
    out = np.full(n, np.nan)
    # rolling forward max via reversed cummax over windows
    for t in range(n - H - 1):
        out[t] = close[t+1:t+H+1].max()
    return out

rows = []
for pair in PAIRS:
    df = pd.read_feather(DATA / f"{pair}-15m.feather")
    df = DataframePopulator().add_indicators(df, dataset_type=DatasetType.MINIMAL)
    close = np.asarray(df.close, dtype=float)
    gm = np.nan_to_num(np.asarray(df.guard_metric, dtype=float))
    bw = np.nan_to_num(np.asarray(df.bb_width, dtype=float))
    cond = (gm > 0.2) & (bw > 0.035)          # the breakout_gbb condition, no gain filter
    for H in HORIZONS:
        fm = fwd_max(close, H)
        g = (fm - close) / close * 100
        g = g[cond & ~np.isnan(g)]
        if len(g) < 100: continue
        tot = g[g > 0].sum()
        top10 = np.percentile(g, 90)
        rows.append(dict(pair=pair.split("_")[0], H=H, n=len(g),
                         mean=g.mean(), median=np.median(g),
                         p75=np.percentile(g,75), p90=top10,
                         p95=np.percentile(g,95), p99=np.percentile(g,99),
                         skew=stats.skew(g), exkurt=stats.kurtosis(g),
                         share_top10=g[g>=top10].sum()/tot*100 if tot>0 else np.nan,
                         share_top25=g[g>=np.percentile(g,75)].sum()/tot*100 if tot>0 else np.nan))
r = pd.DataFrame(rows)
agg = r.groupby("H").apply(lambda s: pd.Series({
    "n": s.n.sum(),
    "mean%": np.average(s["mean"], weights=s.n),
    "median%": np.average(s["median"], weights=s.n),
    "p90%": np.average(s["p90"], weights=s.n),
    "p99%": np.average(s["p99"], weights=s.n),
    "p99/median": np.average(s["p99"], weights=s.n)/np.average(s["median"], weights=s.n),
    "skew": np.average(s["skew"], weights=s.n),
    "excess kurt": np.average(s["exkurt"], weights=s.n),
    "% of gain in top 10%": np.average(s["share_top10"], weights=s.n),
    "% of gain in top 25%": np.average(s["share_top25"], weights=s.n),
}), include_groups=False)
print("Forward MFE from breakout-condition bars (guard_metric>0.2 & bb_width>0.035)")
print("pooled XRP/SOL/LINK, 15m\n")
print(agg.T.to_string(float_format=lambda v: f"{v:.2f}"))
print("\n  Gaussian reference: skew 0, excess kurtosis 0, top 10% would hold ~25-30% of a positive-only sum.")
