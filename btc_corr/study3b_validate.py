"""Study 3b — validate the Study-3 feature ADD/DROP via a multivariate model.

Univariate IC (Study 3) is a screen; this tests whether the recommended feature
set actually improves OOS return prediction on a model that uses INTERACTIONS
(LightGBM). Walk-forward per pair, compare:
  BASELINE (current 24-feature include_list)
  STUDY3   (drop 8 low-IC + add vwap_ratio/close_norm = 18 features)
Deterministic feature generation (DataframePopulator MINIMAL); the only change
between arms is the column set. Not the production NN retrain — a fast screen to
decide if that retrain is worth it.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
from btcdata import load_ohlcv, ALTS_1H
from utils.DataframePopulator import DataframePopulator, DatasetType

TF = "1h"
H = 4
N_FOLDS = 3

BASELINE = ["adx_scaled", "aroonosc_scaled", "atr_norm", "bb_position", "bb_width",
            "cci_scaled", "di_diff_scaled", "ema_fast_norm", "fast_diff", "fastk_scaled",
            "fisher_ss", "cg_ss", "gain_norm", "guard_metric_pos", "guard_metric_neg",
            "macd_pos", "macd_neg", "macdhist_norm", "mfi_scaled", "rsi_scaled",
            "sar_ratio", "spread_ma", "vwap_pos", "vwap_neg"]
DROP = ["adx_scaled", "bb_width", "atr_norm", "mfi_scaled", "macd_pos", "macd_neg",
        "cg_ss", "spread_ma"]
ADD = ["vwap_ratio", "close_norm"]
STUDY3 = [c for c in BASELINE if c not in DROP] + ADD


def lgbm():
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.7, min_child_samples=80,
                             reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)


def features_and_target(pair):
    df = load_ohlcv(pair, TF).reset_index()
    feat = DataframePopulator().add_indicators(df.copy(), DatasetType.MINIMAL)
    logc = np.log(feat["close"].astype(float))
    feat["_y"] = (logc.shift(-H) - logc)
    return feat


def walk_forward(X, y):
    n = len(X); start = int(n * 0.40); block = (n - start) // N_FOLDS
    preds, act = [], []
    for i in range(N_FOLDS):
        te0 = start + i * block
        te1 = start + (i + 1) * block if i < N_FOLDS - 1 else n
        tr1 = te0 - H
        if tr1 < 200:
            continue
        m = lgbm().fit(X.iloc[:tr1], y.iloc[:tr1])
        preds.append(pd.Series(m.predict(X.iloc[te0:te1]), index=X.index[te0:te1]))
        act.append(y.iloc[te0:te1])
    p = pd.concat(preds); a = pd.concat(act)
    return spearmanr(p, a).correlation, float((np.sign(p) == np.sign(a)).mean())


def run():
    rows = []
    for pair in ALTS_1H:
        f = features_and_target(pair).replace([np.inf, -np.inf], np.nan)
        for name, cols in [("BASE", BASELINE), ("S3", STUDY3)]:
            use = [c for c in cols if c in f.columns]
            d = f[use + ["_y"]].dropna()
            ic, da = walk_forward(d[use], d["_y"])
            rows.append({"pair": pair, "set": name, "n_feat": len(use), "IC": ic, "dir": da})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    piv = df.pivot(index="pair", columns="set", values="IC")
    piv["dIC"] = piv["S3"] - piv["BASE"]
    dpv = df.pivot(index="pair", columns="set", values="dir")
    print("===== Study 3b: BASELINE(24) vs STUDY3(18) feature set — OOS walk-forward IC =====")
    print(f"  dropped 8 low-IC: {', '.join(DROP)}")
    print(f"  added: {', '.join(ADD)}\n")
    print("  pair      IC_BASE   IC_S3     dIC     dir_BASE dir_S3")
    for p in piv.index:
        print(f"  {p:5s}   {piv.loc[p,'BASE']:+.4f}  {piv.loc[p,'S3']:+.4f}  {piv.loc[p,'dIC']:+.4f}   "
              f"{dpv.loc[p,'BASE']*100:5.1f}%  {dpv.loc[p,'S3']*100:5.1f}%")
    print(f"\n  mean IC_BASE={piv['BASE'].mean():.4f}  mean IC_S3={piv['S3'].mean():.4f}  "
          f"mean dIC={piv['dIC'].mean():+.4f}  improved={int((piv['dIC']>0).sum())}/{len(piv)}")
    df.to_csv(Path(__file__).parent / "study3b_results.csv", index=False)
