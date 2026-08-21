"""Shared data helpers for the BTC lead-lag study (Binance.US feather data)."""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("/Users/philprice95/projects/freqtrade/user_data/data/binanceus")

# pairs with native 1h data (no ETH at 1h); 4h has full coverage incl ETH
ALTS_1H = ["ETH", "SOL", "LTC", "AVAX", "LINK", "XRP", "BCH", "AAVE", "DOT", "NEAR", "SUI", "ZEC"]
ALTS_4H = ["ETH", "SOL", "LTC", "AVAX", "LINK", "XRP", "BCH", "DOGE", "ADA", "DOT"]


def load_ohlcv(pair: str, tf: str) -> pd.DataFrame:
    """Load OHLCV indexed by date (native Binance.US feather files)."""
    return pd.read_feather(DATA / f"{pair}_USDT-{tf}.feather").set_index("date").sort_index()


def log_returns(pair: str, tf: str) -> pd.Series:
    c = load_ohlcv(pair, tf)["close"]
    return np.log(c).diff().rename(pair)


def hours_per_candle(tf: str) -> int:
    return {"1h": 1, "4h": 4}[tf]
