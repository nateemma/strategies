"""Experiment 3 — relative-strength divergence (beta-spread mean reversion).

beta_t   : rolling 30d OLS slope of alt_ret on btc_ret, lagged 1 bar (causal).
e_t      : alt_ret_t - beta_t*btc_ret_t         (per-bar idiosyncratic return)
signal z : z-score of the trailing 24h cumulative residual (recent divergence)
fwd_e    : idiosyncratic return over [t+1, t+H], beta_t held fixed.

Does strongly negative spread precede outperformance (reversion)?
Strategy: position = -clip(z), beta-hedged, held 1 bar -> equity curve.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
from btcdata import log_returns, ALTS_1H

OUT = Path(__file__).parent
TF = "1h"
W = 720          # beta window (30d @1h)
L = 24           # spread lookback (24h)
HORIZONS = [4, 8, 24]
PERIODS_YR = 24 * 365


def build(alt):
    df = pd.concat([log_returns("BTC", TF).rename("b"),
                    log_returns(alt, TF).rename("a")], axis=1).dropna()
    b, a = df["b"], df["a"]
    beta = (a.rolling(W).cov(b) / b.rolling(W).var()).shift(1)   # causal
    e = a - beta * b
    S = e.rolling(L).sum()
    z = (S - S.rolling(W).mean()) / (S.rolling(W).std() + 1e-9)
    out = pd.DataFrame({"b": b, "a": a, "beta": beta, "e": e, "z": z})
    for H in HORIZONS:
        fa = a.rolling(H).sum().shift(-H)     # sum a[t+1..t+H]
        fb = b.rolling(H).sum().shift(-H)
        out[f"fwd_e_{H}"] = fa - beta * fb
    return out.replace([np.inf, -np.inf], np.nan)


def run():
    rows, pnls, pooled = [], {}, []
    for alt in ALTS_1H:
        d = build(alt).dropna(subset=["z", "e"])
        rec = {"alt": alt}
        for H in HORIZONS:
            m = d.dropna(subset=[f"fwd_e_{H}"])
            rec[f"corr_{H}h"] = round(spearmanr(m["z"], m[f"fwd_e_{H}"]).correlation, 3)
        # beta-hedged contrarian strategy: position = -clip(z), hold 1 bar
        pos = -d["z"].clip(-2, 2)
        pnl = (pos * d["e"].shift(-1)).dropna()
        rec["exp_bp"] = round(pnl.mean() * 1e4, 2)                      # mean bp/bar
        rec["sharpe"] = round(pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(PERIODS_YR), 2)
        rows.append(rec)
        pnls[alt] = pnl.cumsum()
        p = d[["z"]].copy(); p["fwd_e_8"] = d["fwd_e_8"]; p["alt"] = alt
        pooled.append(p.dropna())
    return pd.DataFrame(rows), pnls, pd.concat(pooled)


def conditional_table(pooled):
    pooled = pooled.copy()
    pooled["q"] = pd.qcut(pooled["z"], 5, labels=False, duplicates="drop")
    g = pooled.groupby("q")["fwd_e_8"].agg(["mean", "median", "count"])
    g["mean_bp"] = (g["mean"] * 1e4).round(2)
    return g


if __name__ == "__main__":
    res, pnls, pooled = run()
    res.to_csv(OUT / "exp3_spread_results.csv", index=False)
    print("===== Experiment 3: beta-spread mean reversion (1h) =====")
    print("  corr(spread z, forward idiosyncratic return) — NEGATIVE = reversion\n")
    print(res.to_string(index=False))
    print(f"\n  mean corr_8h={res.corr_8h.mean():+.3f}   "
          f"strategies with positive expectancy={int((res.exp_bp>0).sum())}/{len(res)}   "
          f"mean Sharpe={res.sharpe.mean():.2f}")

    print("\n===== Conditional: forward 8h idiosyncratic return by spread quintile (pooled) =====")
    ct = conditional_table(pooled)
    print("  q0 = most negative spread (recently underperformed BTC-beta)")
    for q, r in ct.iterrows():
        print(f"   Q{int(q)}: mean fwd_e(8h) = {r['mean_bp']:+7.2f} bp   (n={int(r['count'])})")

    # equity curves: combined equal-weight portfolio + top-Sharpe alts
    port = pd.concat([pnls[a].rename(a) for a in pnls], axis=1).ffill().mean(axis=1)
    plt.figure(figsize=(10, 5))
    plt.plot(port.index, port.values, "k", lw=2, label="equal-wt portfolio")
    top = res.sort_values("sharpe", ascending=False).head(4)["alt"]
    for a in top:
        plt.plot(pnls[a].index, pnls[a].values, lw=1, alpha=0.7, label=a)
    plt.axhline(0, color="gray", lw=0.5)
    plt.title("Exp 3: beta-hedged spread-reversion equity (cum. idiosyncratic return)")
    plt.ylabel("cumulative log return"); plt.legend(ncol=3, fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "exp3_equity.png", dpi=110); plt.close()
    print(f"\n  saved exp3_spread_results.csv + exp3_equity.png under {OUT}")
