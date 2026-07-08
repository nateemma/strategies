"""Feature construction. No lookahead: every feature at candle t uses only data
up to and including t. Built symmetrically for the alt and for BTC."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import talib

sys.path.append(str(Path(__file__).parent))
from btcdata import load_ohlcv, hours_per_candle

RET_HOURS = [1, 2, 4, 8, 12, 24]


def build_features(pair: str, tf: str, prefix: str) -> pd.DataFrame:
    df = load_ohlcv(pair, tf)
    hpc = hours_per_candle(tf)
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    logc = np.log(c)
    cv, hv, lv = c.values, h.values, l.values

    F = pd.DataFrame(index=df.index)
    for n in RET_HOURS:                       # multi-scale returns (hours -> candles)
        k = n // hpc
        if k >= 1:
            F[f"{prefix}_ret_{n}h"] = logc - logc.shift(k)
    F[f"{prefix}_rsi14"] = talib.RSI(cv, 14)
    F[f"{prefix}_adx"] = talib.ADX(hv, lv, cv, 14)
    F[f"{prefix}_atrpct"] = talib.ATR(hv, lv, cv, 14) / cv
    w = max(2, 24 // hpc)
    F[f"{prefix}_volz"] = (v - v.rolling(w).mean()) / (v.rolling(w).std() + 1e-9)
    F[f"{prefix}_vol24"] = logc.diff().rolling(w).std()
    F[f"{prefix}_ema20d"] = cv / talib.EMA(cv, 20) - 1.0
    F[f"{prefix}_ema50d"] = cv / talib.EMA(cv, 50) - 1.0
    F[f"{prefix}_brkhi20"] = cv / h.rolling(20).max() - 1.0   # dist to 20-bar high
    F[f"{prefix}_brklo20"] = cv / l.rolling(20).min() - 1.0   # dist above 20-bar low
    return F


def forward_return(pair: str, tf: str, horizon_candles: int) -> pd.Series:
    c = load_ohlcv(pair, tf)["close"].astype(float)
    logc = np.log(c)
    return (logc.shift(-horizon_candles) - logc).rename("y")
