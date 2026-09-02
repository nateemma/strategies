"""Shared data helpers for the market-phase study (Binance.US 15m feather data).

Follows the btc_corr/btcdata.py pattern: __file__-relative, absolute DATA path,
re-runnable from a clean checkout with no manual path editing (FR-006).

Run from the freqtrade root - the system Python has no pandas:

    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/regimedata.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("/Users/philprice95/projects/freqtrade/user_data/data/binanceus")
TF = "15m"
BARS_PER_HOUR = 4

# Pinned evaluation windows (research R3). W1 is the deployment regime and
# carries the headline verdict; W3 is the temporally distant era. W1 and W2 are
# only three months apart and CANNOT establish persistence between them.
WINDOWS = {
    "W1": ("20250901", "20260831"),
    "W2": ("20240601", "20250531"),
    "W3": ("20230101", "20231231"),
}

# Market-breadth coverage rule (research R4). Below this the market-level label
# is UNKNOWN rather than a misleading state computed over a handful of coins.
MIN_PAIRS = 20
MIN_HISTORY_BARS = 200

UNKNOWN = "UNKNOWN"

# The host's traded universe (config/config.json StaticPairList). Asset-level
# instruments are only meaningful here: Phase C aligns phase labels to the
# host's realized trades, which occur nowhere else.
TRADED_PAIRS = ["ZEC", "XRP", "SOL", "LINK", "NEAR", "AAVE",
                "SUI", "AVAX", "LTC", "BCH", "DOT"]

# Market-level instruments are built from a LIQUID universe, not from every
# feather on disk. Measured 2026-09-02: of the 77 pairs with 15m data, 61 have
# >50% zero-volume bars in W1 and 16 exceed 90% - those carry stale
# forward-filled prices with no real trading (OCEAN: 97.1% zero-volume, a
# 3,078-bar flat run). Breadth computed over them measures listing artifacts
# rather than market state, and 480 flat bars drive rolling variance to zero,
# making the estimators genuinely undefined.
#
# The filter is applied over ALL THREE windows so the universe is FIXED across
# eras. A universe that changed definition per window would manufacture
# instability and corrupt the Principle II comparison.
MAX_ZERO_VOLUME_SHARE = 0.80


def liquid_pairs(tf: str = TF) -> list[str]:
    """Pairs active enough for market-level measures in every evaluation window.

    Returns 29 pairs as of 2026-09-02, a superset of TRADED_PAIRS.
    """
    out = []
    for p in available_pairs(tf):
        df = load_ohlcv(p, tf)
        if all((window_slice(df, w)["volume"] == 0).mean() <= MAX_ZERO_VOLUME_SHARE
               if len(window_slice(df, w)) else False
               for w in WINDOWS):
            out.append(p)
    return out


def available_pairs(tf: str = TF) -> list[str]:
    """Every pair with native feather data at the given timeframe, sorted."""
    return sorted(p.name.split("_")[0] for p in DATA.glob(f"*_USDT-{tf}.feather"))


def load_ohlcv(pair: str, tf: str = TF) -> pd.DataFrame:
    """Load OHLCV indexed by a tz-aware UTC date.

    The index dtype matters downstream: the strategy-side merge joins on a
    tz-aware `date` column, and a naive timestamp produces a silent all-NaN
    merge rather than an error.
    """
    df = pd.read_feather(DATA / f"{pair}_USDT-{tf}.feather")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date").sort_index()


def log_returns(pair: str, tf: str = TF) -> pd.Series:
    return np.log(load_ohlcv(pair, tf)["close"]).diff().rename(pair)


def close_panel(pairs: list[str] | None = None, tf: str = TF) -> pd.DataFrame:
    """Wide close-price panel, one column per pair, union of all timestamps.

    Defaults to the LIQUID universe, not every feather on disk - see
    liquid_pairs(). Pass an explicit list to override.

    Columns are NOT forward-filled: a NaN means the pair had no bar at that
    timestamp (not yet listed, or a gap), and the coverage rule below depends on
    being able to tell those apart from real prices.
    """
    pairs = pairs or liquid_pairs(tf)
    cols = {p: load_ohlcv(p, tf)["close"] for p in pairs}
    return _trim_incomplete_tail(pd.DataFrame(cols).sort_index())


def _trim_incomplete_tail(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop trailing bars that are still being formed.

    The last timestamp in the dataset is normally a PARTIAL candle: the
    download stopped mid-bar, so only some pairs have posted. Observed
    2026-08-31 17:00 with 17 of 29 pairs reporting where every preceding bar
    had all 29. Classifying that bar produces a spurious low-coverage UNKNOWN
    at the right-hand edge of W1 - which then reads as a defect.

    Trailing bars reporting fewer pairs than the bar before them are dropped.
    """
    n = panel.notna().sum(axis=1)
    while len(n) > 1 and n.iloc[-1] < n.iloc[-2]:
        panel = panel.iloc[:-1]
        n = n.iloc[:-1]
    return panel


def coverage(panel: pd.DataFrame) -> pd.Series:
    """Number of pairs qualifying for breadth measures at each bar.

    A pair qualifies once it has at least MIN_HISTORY_BARS of prior bars, so a
    freshly listed coin cannot enter a breadth calculation with an undefined
    trend state.
    """
    has_bar = panel.notna()
    history = has_bar.cumsum()
    return (has_bar & (history >= MIN_HISTORY_BARS)).sum(axis=1).rename("coverage_n")


def coverage_ok(panel: pd.DataFrame) -> pd.Series:
    """Boolean mask of bars where breadth measures are defined (research R4)."""
    return (coverage(panel) >= MIN_PAIRS).rename("coverage_ok")


def window_slice(obj, window: str):
    """Restrict a Series/DataFrame to one of the pinned evaluation windows."""
    start, end = WINDOWS[window]
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return obj.loc[(obj.index >= lo) & (obj.index < hi)]


def bars(hours: float) -> int:
    """Convert hours to 15m bars."""
    return int(round(hours * BARS_PER_HOUR))


if __name__ == "__main__":
    print(f"pairs at {TF}: {len(available_pairs())} total, {len(liquid_pairs())} liquid, {len(TRADED_PAIRS)} traded")

    panel = close_panel()
    print(f"panel: {panel.shape[0]} bars x {panel.shape[1]} pairs")
    print(f"span:  {panel.index.min()} -> {panel.index.max()}  (tz={panel.index.tz})")

    cov = coverage(panel)
    ok = coverage_ok(panel)
    first_ok = ok.idxmax() if ok.any() else None
    print(f"\ncoverage rule: >={MIN_PAIRS} pairs with >={MIN_HISTORY_BARS} bars of history")
    print(f"  first bar meeting it: {first_ok}")
    print(f"  bars UNKNOWN:         {int((~ok).sum())} of {len(ok)} ({(~ok).mean():.1%})")

    print("\nper-window coverage (UNKNOWN inside a window is a DEFECT, not a state):")
    for w in WINDOWS:
        wo = window_slice(ok, w)
        wc = window_slice(cov, w)
        flag = "OK" if wo.all() else f"*** {int((~wo).sum())} UNKNOWN BARS ***"
        print(f"  {w} {WINDOWS[w][0]}-{WINDOWS[w][1]}: "
              f"{len(wo)} bars, coverage {int(wc.min())}-{int(wc.max())} pairs  {flag}")
