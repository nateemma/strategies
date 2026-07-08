"""Study 3 — feature predictive-power audit (cross-pair IC vs forward returns).

Complements Debug/DebugAnalyseIndicators (single-pair, vs classification labels):
here we score every MINIMAL feature by its Spearman IC against the H-bar FORWARD
RETURN, per pair, then aggregate — mean IC + cross-pair sign-consistency. Ranks
which features carry robust predictive signal, and flags:
  * DROP candidates  — currently in include_list but weak/unstable IC
  * ADD candidates   — strong+consistent IC but not in include_list
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
from btcdata import load_ohlcv, ALTS_1H
from utils.DataframePopulator import DataframePopulator, DatasetType
from Framework.FeatureNormalizer import FeatureNormalizer

TF = "1h"
H = 4                       # forward-return horizon (bars)
OUT = Path(__file__).parent
RAW = {"open", "high", "low", "close", "volume", "close_safe", "gain"}


def feature_ic_for_pair(pair):
    df = load_ohlcv(pair, TF).reset_index()
    feat = DataframePopulator().add_indicators(df.copy(), DatasetType.MINIMAL)
    logc = np.log(feat["close"].astype(float))
    fwd = (logc.shift(-H) - logc).to_numpy()
    cols = [c for c in feat.columns if feat[c].dtype.kind in "fi" and c not in RAW]
    ics = {}
    for c in cols:
        x = feat[c].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(fwd)
        if m.sum() > 500 and np.nanstd(x[m]) > 0:
            ics[c] = spearmanr(x[m], fwd[m]).correlation
    return pd.Series(ics, name=pair)


def run():
    il = set(getattr(FeatureNormalizer, "include_list", []) or [])
    per_pair = []
    for a in ALTS_1H:
        try:
            per_pair.append(feature_ic_for_pair(a))
        except Exception as e:
            print(f"  (skip {a}: {e})")
    M = pd.concat(per_pair, axis=1)            # features x pairs
    mean_ic = M.mean(axis=1)
    # cross-pair sign consistency: fraction of pairs agreeing with the mean sign
    consistency = M.apply(lambda r: (np.sign(r.dropna()) == np.sign(mean_ic[r.name])).mean(), axis=1)
    res = pd.DataFrame({
        "feature": M.index,
        "in_IL": [f in il for f in M.index],
        "mean_IC": mean_ic.round(4).values,
        "abs_IC": mean_ic.abs().round(4).values,
        "consistency": consistency.round(2).values,
        "std_IC": M.std(axis=1).round(4).values,
    }).sort_values("abs_IC", ascending=False).reset_index(drop=True)
    return res


if __name__ == "__main__":
    res = run()
    res.to_csv(OUT / "study3_feature_ic.csv", index=False)
    print(f"===== Study 3: feature IC vs {H}h forward return (cross-pair, {len(ALTS_1H)} alts) =====\n")
    print("TOP 20 by |mean IC|  (* = in include_list):")
    for _, r in res.head(20).iterrows():
        star = "*" if r["in_IL"] else " "
        print(f"  {star} {r['feature']:18s} IC={r['mean_IC']:+.4f}  consistency={r['consistency']:.2f}  std={r['std_IC']:.4f}")

    drop = res[(res.in_IL) & (res.abs_IC < 0.02)].sort_values("abs_IC")
    add = res[(~res.in_IL) & (res.abs_IC >= 0.03) & (res.consistency >= 0.75)].sort_values("abs_IC", ascending=False)
    print(f"\nDROP candidates (in include_list, |IC|<0.02): "
          f"{', '.join(drop['feature']) if len(drop) else 'none'}")
    print(f"ADD candidates (not in IL, |IC|>=0.03 & consistency>=0.75): "
          f"{', '.join(add['feature']) if len(add) else 'none'}")
    inc = res[res.in_IL]
    print(f"\n  include_list features: {len(inc)}  |  mean |IC| in-IL={inc.abs_IC.mean():.4f} "
          f"vs out-IL={res[~res.in_IL].abs_IC.mean():.4f}")
    print(f"  saved study3_feature_ic.csv under {OUT}")
