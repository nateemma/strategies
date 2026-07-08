"""Study 1 — cross-sectional alt rotation (spot-legal salvage of btc_corr Exp 3).

Each rebalance, rank alts by their recent idiosyncratic (cross-sectionally
de-meaned) L-bar return. Test the forward return of quintile-sorted LONG-ONLY
baskets: does buying the recently-*weakest* alts (reversion) or *strongest*
(momentum) beat an equal-weight-all benchmark? No shorting — you only choose
which alts to hold, so it's spot-tradeable (fits the Basket family).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
from btcdata import log_returns, ALTS_1H

OUT = Path(__file__).parent
TF = "1h"
L = 24                      # signal lookback (bars)
HOLD = [4, 8, 24]           # rebalance/hold horizons
PERIODS_YR = 24 * 365


def returns_matrix():
    cols = {a: log_returns(a, TF) for a in ALTS_1H}
    return pd.DataFrame(cols).dropna(how="all")


def stats(period_rets, per_yr):
    r = np.asarray(period_rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return dict(tot=np.nan, sharpe=np.nan, maxdd=np.nan, n=len(r))
    eq = np.cumsum(r)
    dd = float((eq - np.maximum.accumulate(eq)).min())
    return dict(tot=float(eq[-1]) * 100, sharpe=r.mean() / (r.std() + 1e-12) * np.sqrt(per_yr),
                maxdd=dd * 100, n=len(r))


def run():
    rets = returns_matrix()
    sig = rets.rolling(L).sum()
    sig_dm = sig.sub(sig.mean(axis=1), axis=0)     # cross-sectional de-mean = idiosyncratic
    T = len(rets)
    cond_rows, eq_curves, summary = [], {}, []

    for H in HOLD:
        fwd = rets.rolling(H).sum().shift(-H)       # H-bar fwd return per alt aligned to t
        per_yr = PERIODS_YR / H
        rebal = range(L, T - H, H)                  # non-overlapping holds
        rev, mom, bench = [], [], []
        # 5-bucket conditional forward return (pooled)
        buckets = {q: [] for q in range(5)}
        for t in rebal:
            s = sig_dm.iloc[t].dropna()
            f = fwd.iloc[t]
            if len(s) < 5:
                continue
            ranks = s.rank(pct=True)
            for q in range(5):
                sel = s.index[(ranks > q / 5) & (ranks <= (q + 1) / 5)]
                if len(sel):
                    buckets[q].append(f[sel].mean())
            lo = s.index[ranks <= 0.2]
            hi = s.index[ranks > 0.8]
            rev.append(f[lo].mean()); mom.append(f[hi].mean()); bench.append(f.mean())
        # net of cost: assume full basket turnover each rebalance, 10bp round-trip
        cost = 10e-4
        rev_net = [x - cost for x in rev]; mom_net = [x - cost for x in mom]
        for name, series in [("reversion", rev_net), ("momentum", mom_net), ("benchmark", bench)]:
            st = stats(series, per_yr)
            st["cal"] = st["tot"] / abs(st["maxdd"]) if st["maxdd"] else np.nan
            summary.append({"H": H, "strat": name, **{k: round(v, 3) if isinstance(v, float) else v
                                                       for k, v in st.items()}})
        if H == 8:
            for q in range(5):
                cond_rows.append({"quintile": q, "mean_fwd_bp": round(np.nanmean(buckets[q]) * 1e4, 2),
                                  "n": len(buckets[q])})
            eq_curves = {"reversion": np.cumsum(rev_net), "momentum": np.cumsum(mom_net),
                         "benchmark": np.cumsum(bench)}
    return pd.DataFrame(summary), pd.DataFrame(cond_rows), eq_curves


if __name__ == "__main__":
    summ, cond, eq = run()
    summ.to_csv(OUT / "study1_summary.csv", index=False)
    print("===== Study 1: cross-sectional alt rotation (net @10bp/rebalance) =====")
    print(f"  signal = idiosyncratic {L}-bar return; long-only quintile baskets\n")
    print(summ.to_string(index=False))
    print("\n  Conditional fwd-8h return by signal quintile (Q0=weakest recent, Q4=strongest):")
    for _, r in cond.iterrows():
        print(f"    Q{int(r['quintile'])}: {r['mean_fwd_bp']:+7.2f} bp  (n={int(r['n'])})")

    plt.figure(figsize=(10, 5))
    for name, e in eq.items():
        plt.plot(e, label=name, lw=2 if name != "benchmark" else 1)
    plt.axhline(0, color="gray", lw=0.5)
    plt.title("Study 1: long-only rotation equity (H=8, net @10bp) — cum log return")
    plt.ylabel("cumulative log return"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "study1_equity.png", dpi=110); plt.close()
    print(f"\n  saved study1_summary.csv + study1_equity.png under {OUT}")
