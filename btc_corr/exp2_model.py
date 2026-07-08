"""Experiment 2 — do BTC features improve altcoin-return prediction?

Model A: alt-only features.  Model B: alt + BTC features.
Walk-forward OOS (3 expanding folds, train-on-past). LightGBM regressor.
Metrics: Information Coefficient (Spearman rho), directional accuracy, R2.
Importance: LightGBM gain (aggregated) + OOS permutation importance.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
from btcdata import ALTS_1H, hours_per_candle
from features import build_features, forward_return

OUT = Path(__file__).parent
TF = "1h"
H = 4                       # forward horizon in candles (4h at 1h)
N_FOLDS = 3


def lgbm():
    return lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.02, num_leaves=31,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=80,
        reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1,
    )


def walk_forward(X, y, embargo):
    """Yield (train_idx, test_idx) for N_FOLDS expanding folds over the last 60%."""
    n = len(X)
    start = int(n * 0.40)
    block = (n - start) // N_FOLDS
    for i in range(N_FOLDS):
        te0 = start + i * block
        te1 = start + (i + 1) * block if i < N_FOLDS - 1 else n
        tr1 = te0 - embargo                      # embargo so target window can't leak
        if tr1 < 100:
            continue
        yield np.arange(0, tr1), np.arange(te0, te1)


def eval_model(X, y):
    """Return OOS pred vs actual concatenated across folds + fitted-on-all model."""
    preds, actuals = [], []
    for tr, te in walk_forward(X, y, embargo=H):
        m = lgbm().fit(X.iloc[tr], y.iloc[tr])
        preds.append(pd.Series(m.predict(X.iloc[te]), index=X.index[te]))
        actuals.append(y.iloc[te])
    p = pd.concat(preds); a = pd.concat(actuals)
    ic = spearmanr(p, a).correlation
    diracc = float((np.sign(p) == np.sign(a)).mean())
    r2 = r2_score(a, p)
    return ic, diracc, r2, p, a


def run():
    btcF = build_features("BTC", TF, "btc")
    rows, gain_imp = [], []
    perm_records = {}
    for alt in ALTS_1H:
        altF = build_features(alt, TF, "alt")
        y = forward_return(alt, TF, H)
        A = altF.join(y, how="inner").replace([np.inf, -np.inf], np.nan).dropna()
        B = altF.join(btcF, how="inner").join(y, how="inner").replace(
            [np.inf, -np.inf], np.nan).dropna()
        ya, Xa = A["y"], A.drop(columns="y")
        yb, Xb = B["y"], B.drop(columns="y")

        ic_a, da_a, r2_a, _, _ = eval_model(Xa, ya)
        ic_b, da_b, r2_b, pb, ab = eval_model(Xb, yb)
        rows.append({
            "alt": alt,
            "IC_A": round(ic_a, 4), "IC_B": round(ic_b, 4),
            "dIC": round(ic_b - ic_a, 4),
            "dirA": round(da_a * 100, 1), "dirB": round(da_b * 100, 1),
            "R2_A": round(r2_a, 4), "R2_B": round(r2_b, 4),
        })
        # gain importance from a full-data fit of Model B
        mB = lgbm().fit(Xb, yb)
        gi = pd.Series(mB.booster_.feature_importance(importance_type="gain"),
                       index=Xb.columns)
        gain_imp.append(gi / gi.sum())
        # permutation importance (OOS-ish: last 35% held out) for a few majors
        if alt in ("ETH", "SOL", "LTC"):
            cut = int(len(Xb) * 0.65)
            mtr = lgbm().fit(Xb.iloc[:cut], yb.iloc[:cut])
            pi = permutation_importance(mtr, Xb.iloc[cut:], yb.iloc[cut:],
                                        scoring="r2", n_repeats=5, random_state=0, n_jobs=-1)
            perm_records[alt] = pd.Series(pi.importances_mean, index=Xb.columns)

    res = pd.DataFrame(rows)
    gain = pd.concat(gain_imp, axis=1).mean(axis=1).sort_values(ascending=False)
    return res, gain, perm_records


if __name__ == "__main__":
    res, gain, perm = run()
    res.to_csv(OUT / "exp2_ab_results.csv", index=False)
    print("===== Experiment 2: Model A (alt) vs Model B (alt+BTC), OOS walk-forward =====")
    print(f"  target = {H*hours_per_candle(TF)}h-forward return @ {TF}\n")
    print(res.to_string(index=False))
    print(f"\n  mean IC_A={res.IC_A.mean():.4f}  mean IC_B={res.IC_B.mean():.4f}  "
          f"mean dIC={res.dIC.mean():+.4f}  alts improved={int((res.dIC>0).sum())}/{len(res)}")
    print(f"  mean dirA={res.dirA.mean():.1f}%  mean dirB={res.dirB.mean():.1f}%")

    print("\n===== Aggregated LightGBM gain importance (Model B, top 15) =====")
    btc_ranks = [i for i, k in enumerate(gain.index) if k.startswith("btc_")]
    for rank, (k, val) in enumerate(gain.head(15).items()):
        tag = "  <-- BTC" if k.startswith("btc_") else ""
        print(f"  {rank+1:2d}. {k:16s} {val*100:5.2f}%{tag}")
    btc_share = gain[[k for k in gain.index if k.startswith('btc_')]].sum()
    print(f"\n  BTC features = {btc_share*100:.1f}% of total gain importance; "
          f"top BTC feature ranks at #{btc_ranks[0]+1}")

    # bar plot of gain importance
    plt.figure(figsize=(8, 7))
    g = gain.head(20)[::-1]
    colors = ["tab:orange" if k.startswith("btc_") else "tab:blue" for k in g.index]
    plt.barh(range(len(g)), g.values * 100, color=colors)
    plt.yticks(range(len(g)), g.index, fontsize=8)
    plt.xlabel("mean gain importance (%)")
    plt.title(f"Model B feature importance ({H*hours_per_candle(TF)}h fwd, {TF})\norange = BTC")
    plt.tight_layout(); plt.savefig(OUT / "exp2_importance.png", dpi=110); plt.close()
    print(f"\n  saved exp2_ab_results.csv + exp2_importance.png under {OUT}")
