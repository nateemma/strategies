"""035 synthetic fixtures: two-plus pairs of 15-min OHLCV with gap-filled bars, matching global
klines (ms stamps before 2025, us after), Binance.US klines and 5-min metrics starting mid-span."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ohlcv(n: int, seed: int, start="2024-11-01", gap_every: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.002, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.002, n))
    v = rng.uniform(10, 100, n)
    df = pd.DataFrame({"date": dates, "open": o, "high": h, "low": lo, "close": c, "volume": v})
    if gap_every:
        g = np.arange(n) % gap_every == gap_every - 1
        prev = df["close"].shift(1).to_numpy()
        for col in ("open", "high", "low", "close"):
            df.loc[g, col] = prev[g]
        df.loc[g, "volume"] = 0.0
    return df


def _klines(ohlcv: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(ohlcv)
    vol = rng.uniform(50, 500, n)
    trades = rng.integers(20, 400, n)
    return pd.DataFrame({
        "open_time": ohlcv["date"].to_numpy(), "open": ohlcv["open"], "high": ohlcv["high"],
        "low": ohlcv["low"], "close": ohlcv["close"], "volume": vol,
        "quote_volume": vol * ohlcv["close"].to_numpy(), "trades": trades,
        "taker_buy_base": vol * rng.uniform(0.3, 0.7, n),
        "taker_buy_quote": vol * ohlcv["close"].to_numpy() * 0.5})


def _metrics(start, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    oi = 1e5 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    return pd.DataFrame({"create_time": t, "sum_open_interest": oi,
                         "sum_open_interest_value": oi * 100,
                         "count_toptrader_long_short_ratio": rng.uniform(0.8, 2.0, n),
                         "sum_toptrader_long_short_ratio": rng.uniform(0.8, 2.0, n),
                         "count_long_short_ratio": rng.uniform(0.8, 2.0, n),
                         "sum_taker_long_short_vol_ratio": rng.uniform(0.5, 2.0, n)})


@pytest.fixture(scope="session")
def synth():
    """Sources for 7 pairs + BTC, 4000 bars (crosses 2025-01-01, where archive stamps become us)."""
    n = 4000
    pairs = ["P0/USDT", "P1/USDT", "P2/USDT", "P3/USDT", "P4/USDT", "P5/USDT", "P6/USDT"]
    ohlcv = {p: _ohlcv(n, i, gap_every=(7 if i == 0 else 0)) for i, p in enumerate(pairs)}
    btc = _ohlcv(n, 99)
    glob = {p: _klines(ohlcv[p], 100 + i) for i, p in enumerate(pairs)}
    us = {p: _klines(ohlcv[p], 200 + i) for i, p in enumerate(pairs)}
    for p in pairs:  # Binance.US has no row where the venue bar is gap-filled
        us[p] = us[p][ohlcv[p]["volume"].to_numpy() > 0].reset_index(drop=True)
    metrics = {p: _metrics("2024-12-01", (n - 2880) * 3, 300 + i) for i, p in enumerate(pairs)}
    prem = {p: _klines(ohlcv[p], 400 + i).assign(close=lambda d: d["close"] * 0 + np.sin(
        np.arange(len(d)) / 50) * 1e-4) for i, p in enumerate(pairs)}
    return {"pairs": pairs, "ohlcv": ohlcv, "btc": btc, "global": glob, "us": us,
            "metrics": metrics, "premium": prem}
