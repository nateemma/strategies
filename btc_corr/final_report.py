"""Consolidate Experiments 1-4 into a per-altcoin master table, ranked by
out-of-sample predictive power (Exp-2 walk-forward IC)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from btcdata import log_returns, ALTS_1H
from exp3_spread import build, PERIODS_YR
from exp4_regime import btc_regime, ORDER

OUT = Path(__file__).parent

exp2 = pd.read_csv(OUT / "exp2_ab_results.csv").set_index("alt")

btc_ret = log_returns("BTC", "1h")
reg = btc_regime()
rows = []
for a in ALTS_1H:
    alt_ret = log_returns(a, "1h")
    lag0 = pd.concat([btc_ret.rename("b"), alt_ret.rename("a")], axis=1).dropna()
    lag0_corr = round(float(lag0["b"].corr(lag0["a"])), 3)
    d = build(a).dropna(subset=["z", "e"])
    pos = -d["z"].clip(-2, 2) / 2.0
    gross = pos * d["e"].shift(-1)
    turn = pos.diff().abs()
    net5 = (gross - 5e-4 * 2 * turn).dropna()
    net10 = (gross - 10e-4 * 2 * turn).dropna()

    # per-alt directional return by BTC regime
    r1 = log_returns(a, "1h").shift(-1)
    g = pd.DataFrame({"reg": reg, "r": r1}).dropna().groupby("reg")["r"].mean() * 1e4
    g = g.reindex(ORDER)
    best_reg, best_bp = g.idxmax(), g.max()

    rows.append({
        "alt": a,
        "lag0_corr": lag0_corr,
        "IC_B": exp2.loc[a, "IC_B"],
        "dIC": exp2.loc[a, "dIC"],
        "spread_net5_bp": round(net5.mean() * 1e4, 2),
        "spread_net10_bp": round(net10.mean() * 1e4, 2),
        "spread_sharpe5": round(net5.mean() / (net5.std() + 1e-12) * np.sqrt(PERIODS_YR), 2),
        "best_regime": best_reg,
        "best_regime_bp": round(float(best_bp), 2),
    })

master = pd.DataFrame(rows).sort_values("IC_B", ascending=False).reset_index(drop=True)
master.index += 1
master.to_csv(OUT / "final_master_table.csv")

pd.set_option("display.width", 200)
print("===== MASTER TABLE — ranked by OOS predictive power (Exp-2 walk-forward IC) =====\n")
print(master.to_string())

shortlist = master[(master.IC_B > 0.05) & (master.spread_net10_bp > 0)]
print("\n===== TRADEABLE SHORTLIST (OOS IC>0.05 AND spread survives 10bp costs) =====")
print("  " + ", ".join(shortlist["alt"].tolist()))
print(f"\n  mean lag0 corr (all)     : {master.lag0_corr.mean():.3f}  (contemporaneous; lead-lag ~0)")
print(f"  mean OOS IC_B            : {master.IC_B.mean():.4f}")
print(f"  mean dIC (BTC feature lift): {master.dIC.mean():+.4f}  ({int((master.dIC>0).sum())}/{len(master)} improved)")
print(f"  spread survives 10bp     : {int((master.spread_net10_bp>0).sum())}/{len(master)}")
