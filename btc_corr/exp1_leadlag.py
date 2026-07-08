"""Experiment 1 — BTC->alt lead-lag correlation.

corr(btc_ret_t, alt_ret_(t+lag)) for lag = 0..24 HOURS. Positive lag means the
alt lags BTC (BTC leads). Full-sample peak corr/lag + rolling 90-day stability.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
from btcdata import log_returns, hours_per_candle, ALTS_1H, ALTS_4H

OUT = Path(__file__).parent
MAX_LAG_H = 24


def lag_curve(btc: pd.Series, alt: pd.Series, tf: str):
    """Return (lags_hours, corrs). corr(btc[t], alt[t+lag])."""
    hpc = hours_per_candle(tf)
    df = pd.concat([btc.rename("b"), alt.rename("a")], axis=1).dropna()
    lags_h, corrs = [], []
    for h in range(0, MAX_LAG_H + 1, hpc):          # only integer-candle lags
        k = h // hpc
        c = df["b"].corr(df["a"].shift(-k))          # alt[t+k] aligned to btc[t]
        lags_h.append(h)
        corrs.append(c)
    return np.array(lags_h), np.array(corrs)


def rolling_stability(btc, alt, tf, win_days=90, step_days=30):
    """Peak corr + peak lag per rolling window."""
    hpc = hours_per_candle(tf)
    win = int(win_days * 24 / hpc)
    step = int(step_days * 24 / hpc)
    df = pd.concat([btc.rename("b"), alt.rename("a")], axis=1).dropna()
    rows = []
    for start in range(0, len(df) - win, step):
        w = df.iloc[start:start + win]
        best_c, best_h = -2, None
        for h in range(0, MAX_LAG_H + 1, hpc):
            k = h // hpc
            c = w["b"].corr(w["a"].shift(-k))
            if c is not None and c > best_c:
                best_c, best_h = c, h
        rows.append((w.index[0], best_h, best_c))
    return pd.DataFrame(rows, columns=["window_start", "peak_lag_h", "peak_corr"])


def run(tf, alts):
    btc = log_returns("BTC", tf)
    summary, curves, stab = [], {}, {}
    for a in alts:
        alt = log_returns(a, tf)
        lags, corrs = lag_curve(btc, alt, tf)
        peak_i = int(np.nanargmax(corrs))
        curves[a] = (lags, corrs)
        rs = rolling_stability(btc, alt, tf)
        stab[a] = rs
        summary.append({
            "alt": a, "tf": tf,
            "peak_corr": round(float(corrs[peak_i]), 3),
            "peak_lag_h": int(lags[peak_i]),
            "corr_lag0": round(float(corrs[0]), 3),
            "n_windows": len(rs),
            "roll_peak_lag_median": float(rs["peak_lag_h"].median()),
            "roll_peak_lag_std": round(float(rs["peak_lag_h"].std()), 2),
            "roll_peak_corr_mean": round(float(rs["peak_corr"].mean()), 3),
            "roll_peak_corr_min": round(float(rs["peak_corr"].min()), 3),
            "pct_windows_lag_le_1candle": round(
                float((rs["peak_lag_h"] <= hours_per_candle(tf)).mean()) * 100, 0),
        })
    return pd.DataFrame(summary), curves, stab


def plot_curves(curves, tf):
    plt.figure(figsize=(9, 5))
    for a, (lags, corrs) in curves.items():
        plt.plot(lags, corrs, marker="o", ms=3, label=a)
    plt.axhline(0, color="gray", lw=0.5)
    plt.xlabel("lag (hours): corr(BTC_ret_t, alt_ret_{t+lag})")
    plt.ylabel("Pearson correlation")
    plt.title(f"BTC->alt lead-lag correlation ({tf})")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"exp1_leadlag_{tf}.png", dpi=110)
    plt.close()


def plot_stability(stab, tf):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for a, rs in stab.items():
        ax[0].plot(rs["window_start"], rs["peak_corr"], marker=".", ms=3, label=a)
        ax[1].plot(rs["window_start"], rs["peak_lag_h"], marker=".", ms=3)
    ax[0].set_ylabel("rolling peak corr"); ax[0].grid(alpha=0.3)
    ax[0].legend(ncol=3, fontsize=7); ax[0].set_title(f"90-day rolling stability ({tf})")
    ax[1].set_ylabel("rolling peak lag (h)"); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"exp1_stability_{tf}.png", dpi=110)
    plt.close()


if __name__ == "__main__":
    all_summary = []
    for tf, alts in [("1h", ALTS_1H), ("4h", ALTS_4H)]:
        s, curves, stab = run(tf, alts)
        plot_curves(curves, tf)
        plot_stability(stab, tf)
        s.to_csv(OUT / f"exp1_summary_{tf}.csv", index=False)
        all_summary.append(s)
        print(f"\n===== Experiment 1: lead-lag ({tf}) =====")
        cols = ["alt", "peak_corr", "peak_lag_h", "corr_lag0",
                "roll_peak_lag_median", "roll_peak_lag_std",
                "roll_peak_corr_mean", "roll_peak_corr_min", "pct_windows_lag_le_1candle"]
        print(s[cols].to_string(index=False))
    print(f"\nplots + csv saved under {OUT}")
