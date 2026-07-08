"""Study 2 — exit / holding-period analysis.

For an entry signal, average the forward CUMULATIVE return path over h=1..48 bars
across all entry events. The peak of that path = the optimal holding period; where
it turns over = where the edge decays (and you should exit). Run for two spot
signals: the Study-1 cross-sectional reversion (bottom quintile) and a plain
RSI<30 dip. (Follow-up: feed actual NN-strategy entry timestamps.)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import talib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
from btcdata import load_ohlcv, log_returns, ALTS_1H

OUT = Path(__file__).parent
TF = "1h"
L = 24
MAXH = 48


def returns_matrix():
    return pd.DataFrame({a: log_returns(a, TF) for a in ALTS_1H}).dropna(how="all")


def fwd_path(rets, mask):
    """Mean forward cumulative return (bp) at each horizon h=1..MAXH over masked entries."""
    path = []
    for h in range(1, MAXH + 1):
        fwd_h = rets.rolling(h).sum().shift(-h)      # sum ret[t+1..t+h]
        vals = fwd_h.values[mask.values]
        path.append(np.nanmean(vals) * 1e4)
    return np.array(path)


def run():
    rets = returns_matrix()

    # signal A: cross-sectional reversion — bottom quintile of idiosyncratic L-bar return
    sig = rets.rolling(L).sum()
    sig_dm = sig.sub(sig.mean(axis=1), axis=0)
    ranks = sig_dm.rank(axis=1, pct=True)
    mask_xsec = ranks <= 0.2

    # signal B: RSI(14) < 30 dip, per alt (index-aligned Series -> DataFrame)
    rsi_series = {}
    for a in ALTS_1H:
        df = load_ohlcv(a, TF)
        rsi_series[a] = pd.Series(talib.RSI(df["close"].values, 14), index=df.index)
    rsi = pd.DataFrame(rsi_series).reindex(rets.index)
    mask_rsi = rsi < 30

    paths = {
        "xsec_reversion (bottom Q)": fwd_path(rets, mask_xsec.reindex_like(rets).fillna(False)),
        "RSI<30 dip": fwd_path(rets, mask_rsi.reindex_like(rets).fillna(False)),
    }
    return paths


if __name__ == "__main__":
    paths = run()
    hs = np.arange(1, MAXH + 1)
    print("===== Study 2: forward cumulative-return path after entry (bp) =====\n")
    for name, p in paths.items():
        peak_h = int(hs[np.nanargmax(p)])
        print(f"  {name}:")
        print(f"    peak at h={peak_h}  ({p[peak_h-1]:+.2f} bp);  "
              f"h4={p[3]:+.2f}  h8={p[7]:+.2f}  h16={p[15]:+.2f}  h24={p[23]:+.2f}  h48={p[47]:+.2f} bp")
    plt.figure(figsize=(10, 5))
    for name, p in paths.items():
        plt.plot(hs, p, marker=".", ms=3, label=name)
        pk = int(np.nanargmax(p))
        plt.scatter([hs[pk]], [p[pk]], s=60, zorder=5)
    plt.axhline(0, color="gray", lw=0.5)
    plt.xlabel("holding period h (bars/hours)")
    plt.ylabel("mean forward cumulative return (bp)")
    plt.title("Study 2: edge vs holding period — peak = optimal hold, downturn = exit")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "study2_holding.png", dpi=110); plt.close()
    print(f"\n  saved study2_holding.png under {OUT}")
