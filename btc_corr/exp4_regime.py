"""Experiment 4 — BTC regime detection and altcoin performance by regime.

Regime (causal): ADX strength x EMA-slope direction ->
  Sideways (ADX<20), Weak/Strong Up, Weak/Strong Down.
Measures per regime:
  (a) passive equal-weight alt basket: win rate, avg return, max drawdown;
  (b) the Exp-3 beta-spread reversion strategy net P&L (regime dependence).
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
from exp3_spread import build

OUT = Path(__file__).parent
TF = "1h"
ORDER = ["Strong Down", "Weak Down", "Sideways", "Weak Up", "Strong Up"]


def btc_regime():
    df = load_ohlcv("BTC", TF)
    c, h, l = df["close"].values, df["high"].values, df["low"].values
    ema = talib.EMA(c, 50)
    slope = pd.Series(ema, index=df.index).pct_change(24)     # 24h slope, causal
    adx = pd.Series(talib.ADX(h, l, c, 14), index=df.index)

    reg = pd.Series("Sideways", index=df.index, dtype=object)
    up, dn = slope > 0, slope < 0
    reg[(adx >= 20) & (adx < 30) & up] = "Weak Up"
    reg[(adx >= 30) & up] = "Strong Up"
    reg[(adx >= 20) & (adx < 30) & dn] = "Weak Down"
    reg[(adx >= 30) & dn] = "Strong Down"
    reg[adx < 20] = "Sideways"
    reg[adx.isna() | slope.isna()] = np.nan
    return reg.rename("regime")


def max_drawdown(logret_series):
    eq = logret_series.cumsum()
    return float((eq - eq.cummax()).min())          # log drawdown


def run():
    reg = btc_regime()
    # equal-weight alt basket, next-bar log return
    alt_ret = pd.concat([log_returns(a, TF).rename(a) for a in ALTS_1H], axis=1)
    basket = alt_ret.mean(axis=1)
    fwd1 = basket.shift(-1)                          # next-bar basket return

    # Exp-3 spread strategy portfolio pnl (net @5bp) per bar
    pnls = []
    for a in ALTS_1H:
        d = build(a).dropna(subset=["z", "e"])
        pos = -d["z"].clip(-2, 2) / 2.0
        gross = pos * d["e"].shift(-1)
        cost = 5e-4 * 2 * pos.diff().abs()
        pnls.append((gross - cost).rename(a))
    spread_port = pd.concat(pnls, axis=1).mean(axis=1)

    df = pd.DataFrame({"regime": reg, "fwd1": fwd1, "spread": spread_port}).dropna(subset=["regime"])

    rows = []
    for r in ORDER:
        m = df[df["regime"] == r]
        b = m["fwd1"].dropna()
        s = m["spread"].dropna()
        rows.append({
            "regime": r,
            "pct_time": round(len(m) / len(df) * 100, 1),
            "basket_avg_bp": round(b.mean() * 1e4, 2),
            "basket_winrate": round((b > 0).mean() * 100, 1),
            "basket_maxDD_%": round(max_drawdown(b) * 100, 1),
            "spread_avg_bp": round(s.mean() * 1e4, 2),
            "spread_winrate": round((s > 0).mean() * 100, 1),
        })
    return pd.DataFrame(rows), df


if __name__ == "__main__":
    res, df = run()
    res.to_csv(OUT / "exp4_regime_results.csv", index=False)
    print("===== Experiment 4: altcoin performance by BTC regime (1h) =====\n")
    print(res.to_string(index=False))

    # annualized-ish context
    print("\n  (basket_avg_bp = mean next-bar equal-weight alt return; "
          "spread = Exp-3 reversion net@5bp)")

    # bar plots
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = range(len(res))
    ax[0].bar(x, res["basket_avg_bp"], color=["#b22", "#d77", "#999", "#7d7", "#2b2"])
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels(res["regime"], rotation=30, fontsize=8)
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_ylabel("mean next-bar basket return (bp)")
    ax[0].set_title("Passive long alt basket by BTC regime")
    ax[1].bar(x, res["spread_avg_bp"], color="tab:purple")
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels(res["regime"], rotation=30, fontsize=8)
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylabel("spread strat net (bp/bar)")
    ax[1].set_title("Spread-reversion strategy by BTC regime")
    plt.tight_layout(); plt.savefig(OUT / "exp4_regime.png", dpi=110); plt.close()
    print(f"\n  saved exp4_regime_results.csv + exp4_regime.png under {OUT}")
