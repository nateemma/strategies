"""Signal check: do market-session flags carry predictive signal?

DebugAnalyseIndicators only surveys columns DataframePopulator produces; the
proposed session flags don't exist there yet. This wrapper reuses that tool's
exact per-feature metrics + the production label config (type-17 gbb, min_gain
0.007, HORIZON 48) but injects the temporal/session features first, and scores
them alongside a few known include_list features as reference baselines.

Generalizes beyond session flags: drop any candidate feature into
add_temporal_features (or compute it before analyse_feature) to get its
predictive-signal / modelability score against the production labels BEFORE
committing to a scaler+GAN+classifier retrain.

Conclusion (2026-06-29): session flags scored 0.01–0.11 across ZEC/SOL/XRP/NEAR
— same marginal band as the already-dropped hour_sin/cos, 3–10x below kept
features (bb_width ~0.41). Not adopted.

Run from the freqtrade repo root:
    .venv/bin/python user_data/strategies/Debug/DebugSessionFeatures.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

STRAT = Path("user_data/strategies")
sys.path.insert(0, str(STRAT))
sys.path.insert(0, str(STRAT / "Debug"))

import DebugAnalyseIndicators as D  # noqa: E402
from utils.DataframePopulator import DataframePopulator, DatasetType  # noqa: E402
from Framework.TrainingSignals import (  # noqa: E402
    get_train_buy_signals,
    get_train_sell_signals,
)

# ---- match production labels (TrainingConfig: type 17, 0.007, H=48) ----
LABEL_METHOD = 17
MIN_GAIN = 0.007
HORIZON = 48
DATA_DIR = STRAT.parent / "data" / "binanceus"
PAIRS = ["ZEC_USDT", "SOL_USDT", "XRP_USDT", "NEAR_USDT"]

TEMPORAL = [
    "hour_sin", "hour_cos",           # the cyclic encodings that were dropped before
    "is_weekend",
    "is_asia_open", "is_eu_open", "is_us_open", "is_market_open",
]
# reference features already in include_list — context for "is the session
# signal comparable to features we keep?"
REFERENCE = ["rsi_scaled", "bb_width", "macd_pos", "guard_metric_pos"]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    idx_local = (
        idx.tz_localize("UTC").tz_convert(ZoneInfo("Europe/Paris"))
        if idx.tz is None
        else idx.tz_convert(ZoneInfo("Europe/Paris"))
    )
    hour = idx_local.hour + idx_local.minute / 60.0
    dow = idx_local.dayofweek                      # 0=Mon … 6=Sun
    is_weekend = (dow >= 5).astype(float)
    weekday = 1.0 - is_weekend
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["is_weekend"] = is_weekend
    df["is_asia_open"] = ((hour >= 1) & (hour < 9)).astype(float) * weekday
    df["is_eu_open"] = ((hour >= 9) & (hour < 17.5)).astype(float) * weekday
    df["is_us_open"] = ((hour >= 14.5) & (hour < 21)).astype(float) * weekday
    df["is_market_open"] = (
        (df["is_asia_open"] + df["is_eu_open"] + df["is_us_open"]) > 0
    ).astype(float)
    return df


for pair in PAIRS:
    print("=" * 72)
    print(pair)
    print("=" * 72)
    df = D.load_pair_data(DATA_DIR, pair, "15m")
    df = DataframePopulator().add_indicators(df, dataset_type=DatasetType.MINIMAL)
    df = add_temporal_features(df)

    params = {"horizon": HORIZON, "min_gain": MIN_GAIN, "min_loss": MIN_GAIN}
    buy = get_train_buy_signals(df, method=LABEL_METHOD, params=params).astype(float).values
    sell = get_train_sell_signals(df, method=LABEL_METHOD, params=params).astype(float).values
    print(f"  rows {len(df)}   buy {int(buy.sum())} ({buy.mean()*100:.1f}%)   "
          f"sell {int(sell.sum())} ({sell.mean()*100:.1f}%)")

    rows = []
    for col in TEMPORAL + REFERENCE:
        if col not in df.columns:
            print(f"  (missing column: {col})")
            continue
        rows.append(
            D.analyse_feature(
                col, df[col].astype(float).values, buy, sell,
                already_included=(col in REFERENCE),
            )
        )
    D.print_full_table(rows)
    print()
