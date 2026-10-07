# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore

"""_raw8_features — the 8 causal raw-OHLCV features shared by NNNC_ForwardMAE_Raw
(the strategy) and CreateScalersRaw8 (its scaler builder), so the two cannot
drift apart.

Semantics copied verbatim from ``research/clfstudy/data.py::raw8`` (research
R11) -- causal only, no centred/forward-looking windows:

    ret1      = log(close / close.shift(1))
    ret4      = log(close / close.shift(4))
    log_hc    = log(high / close)
    log_lc    = log(low / close)
    log_oc    = log(open / close)
    range     = (high - low) / close
    logvol_z  = 96-bar causal z-score of log(volume + 1)
    vol_ratio = log((volume + 1) / (rolling_mean(volume, 96) + 1))
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW8_ROLL_WINDOW = 96

RAW8_COLUMNS = [
    "ret1",
    "ret4",
    "log_hc",
    "log_lc",
    "log_oc",
    "range",
    "logvol_z",
    "vol_ratio",
]


def add_raw8_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add the 8 causal raw8 columns to a copy of ``dataframe``.

    ``dataframe`` must already have open/high/low/close/volume columns.
    """
    df = dataframe.copy()
    close = df["close"]
    df["ret1"] = np.log(close / close.shift(1))
    df["ret4"] = np.log(close / close.shift(4))
    df["log_hc"] = np.log(df["high"] / close)
    df["log_lc"] = np.log(df["low"] / close)
    df["log_oc"] = np.log(df["open"] / close)
    df["range"] = (df["high"] - df["low"]) / close

    logvol = np.log(df["volume"] + 1.0)
    roll_mean = logvol.rolling(RAW8_ROLL_WINDOW, min_periods=RAW8_ROLL_WINDOW).mean()
    roll_std = logvol.rolling(RAW8_ROLL_WINDOW, min_periods=RAW8_ROLL_WINDOW).std()
    df["logvol_z"] = (logvol - roll_mean) / roll_std

    vol_mean = df["volume"].rolling(RAW8_ROLL_WINDOW, min_periods=RAW8_ROLL_WINDOW).mean()
    df["vol_ratio"] = np.log((df["volume"] + 1.0) / (vol_mean + 1.0))

    df[RAW8_COLUMNS] = df[RAW8_COLUMNS].fillna(0.0)
    return df
