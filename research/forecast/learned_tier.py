"""038 tiers 2-3: LightGBM walk-forward on the N-candle log return (Stories 4-5; research R7).

Per fold and N: one model pooled over the 10 pairs, trained anchored from 2021-01-01 up to
test_start - (N + 1) candles (labels end before the test window: purge >= N), training rows at
stride 4; rows need eligible_N, real data and data_ok at t, and finite features and target. The f14
model also forecasts the forward window (descriptive). `--set null` trains set L on targets permuted
within each training span (the shuffled-target control, FR-011).

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.forecast.learned_tier --set L [--seed 2]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from research.forecast import common as C
from research.forecast import features as X
from research.forecast import frames as F

TRAIN_STRIDE = 4
BAR = pd.Timedelta(minutes=15)


def train_mask(dates: pd.Series, test_start: pd.Timestamp, n: int) -> np.ndarray:
    """Rows whose N-candle label closes before the test window: date <= test_start - (n+1) bars."""
    return (pd.to_datetime(dates) <= test_start - (n + 1) * BAR).to_numpy()


def null_targets(y: np.ndarray, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(y)


def lw_inputs() -> list[str]:
    t = C.HERE / "truncation.json"
    if not t.exists():
        raise SystemExit("REFUSED: run research.forecast.truncation before --set LW (FR-012)")
    return [c for c, ok in json.loads(t.read_text())["inputs"].items() if ok]


def pooled(fset: str) -> tuple[pd.DataFrame, list[str]]:
    """One pooled frame: pair, date, row, fold, data flags, targets and the set's features."""
    lw = lw_inputs() if fset == "LW" else None
    parts, feat_cols = [], None
    for pair in C.pairs():
        fr = F.read(pair, ["date", "fold", "lc", "volume", "has_data", "data_ok",
                           *[f"target_{n}" for n in C.HORIZONS], *[f"eligible_{n}" for n in C.HORIZONS]])
        f = X.build(pair, fr, fset, lw)
        feat_cols = list(f.columns)
        keep = fr["date"] >= pd.Timestamp("2021-01-01", tz="UTC")
        parts.append(pd.concat([fr.drop(columns=["lc", "volume"]), f], axis=1)[keep]
                     .assign(pair=pair, row=lambda d: d.index.to_numpy()))
    return pd.concat(parts, ignore_index=True), feat_cols


def lgb_params(pr: dict, seed: int) -> dict:
    p = {k: v for k, v in pr["methods"]["lightgbm_params"].items() if k != "training"}
    p["seed"] = seed
    p["verbose"] = -1
    return p


def run(fset: str, seed: int) -> pd.DataFrame:
    from lightgbm import LGBMRegressor

    pr = C.prereg()
    data, cols = pooled(fset)
    finite = np.isfinite(data[cols].to_numpy(np.float32)).all(axis=1)
    base_ok = data["has_data"].to_numpy() & data["data_ok"].to_numpy() & finite
    stride_ok = (data["row"].to_numpy() % TRAIN_STRIDE) == 0
    folds = C.folds()
    out = data.loc[data["fold"] != "", ["pair", "date", "fold"]].copy()
    for n in C.HORIZONS:
        out[f"f_{n}"] = np.nan
    feats = data[cols].to_numpy(np.float32)
    dates = data["date"]
    t0 = time.time()
    for k, (fid, (start, end)) in enumerate(folds.items()):
        pred_mask = ((data["fold"] == fid) | ((data["fold"] == "fwd") & (fid == "f14"))).to_numpy() & base_ok
        for n in C.HORIZONS:
            y_all = data[f"target_{n}"].to_numpy(float)
            tm = (train_mask(dates, start, n) & base_ok & stride_ok & data[f"eligible_{n}"].to_numpy()
                  & np.isfinite(y_all))
            y = y_all[tm]
            if fset == "null":
                y = null_targets(y, seed=seed * 1000 + k * 10 + C.HORIZONS.index(n))
            m = LGBMRegressor(**lgb_params(pr, seed)).fit(feats[tm], y)
            out.loc[data.index[pred_mask], f"f_{n}"] = m.predict(feats[pred_mask])
            print(f"  {fset} s{seed} {fid} n={n:3d} train {tm.sum():7d} predict {pred_mask.sum():6d} "
                  f"[{time.time() - t0:7.1f}s]", flush=True)
    return out.drop(columns=["fold"]).assign(failed=False, skipped_gap=lambda d: d["f_1"].isna())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["L", "LA", "LW", "null"])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    rc = C.refuse_unless_registered()
    if rc:
        return rc
    name = f"lgb_{a.set}" + ("" if a.seed == 1 else f"_s{a.seed}")
    p = run(a.set, a.seed)
    C.write_feather(p.reset_index(drop=True), C.PRED / f"{name}.feather")
    print(f"wrote {name}: {len(p)} rows, forecast share {p['f_1'].notna().mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
