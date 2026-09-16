"""Causal regime features, per pair and for BTC (T005, FR-004).

Deliberately THREE features per block, not a battery. The sparse weighting decides which
matter, and a wide indicator battery is the thing this repository has repeatedly found adds
nothing -- an 85-indicator battery added exactly zero to a triple-barrier target.

CAUSALITY. Row t is a function of bars <= t only. Every statistic is a trailing rolling
window ending at t, and bar t's own close is legitimately known at t. Warm-up rows are
DROPPED, never back-filled: a back-fill would carry a later window's value backwards, which is
a leak that looks like tidiness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regime import regimedata as rd                      # noqa: E402

WINDOW = 96                    # one day at 15-minute bars
BARS_PER_YEAR = 96 * 365

# Calibration target for the jump penalty, PRE-REGISTERED and frozen (research R5).
#
# Expressed as switches PER YEAR, not as a per-bar rate, because the per-bar number is
# meaningless at a glance and that is exactly how a wrong one survives: a first pass targeting
# a 0.01 per-bar rate asked for 184 regime changes a year, which is not a regime model at all.
# The source study's selling point is LOW turnover (44% vs an HMM's 141%); six switches a year
# is a mean regime length near two months, which is what a low-turnover regime model looks like.
TARGET_SWITCHES_PER_YEAR = 6.0
TARGET_SWITCH_RATE = TARGET_SWITCHES_PER_YEAR / BARS_PER_YEAR
FEATURES = ("rvol_realised", "rvol_downside", "sign_persistence")


def _from_close(close: pd.Series, window: int = WINDOW) -> pd.DataFrame:
    r = np.log(close).diff()
    neg = r.where(r < 0, 0.0)
    sgn = np.sign(r)
    return pd.DataFrame({
        "rvol_realised": r.rolling(window).std(),
        "rvol_downside": neg.rolling(window).std(),
        # agreement between consecutive return signs: +1 trending, -1 alternating
        "sign_persistence": (sgn * sgn.shift(1)).rolling(window).mean(),
    }, index=close.index)


def build(pair: str, lo: pd.Timestamp, hi: pd.Timestamp,
          window: int = WINDOW, with_btc: bool = True) -> pd.DataFrame:
    """Per-pair features, plus the same three computed on BTC as a market-wide block.

    The slice is applied AFTER the rolling statistics so the first usable row of the range
    carries a full window of genuine history rather than a truncated one. Slicing first would
    quietly change the feature definition at every fold boundary.
    """
    sym = pair.split("/")[0]
    close = rd.load_ohlcv(sym)["close"]
    out = _from_close(close, window)

    # For BTC itself the market-wide block would DUPLICATE the pair block exactly, giving the
    # sparse weighting perfectly collinear columns to split weight across arbitrarily. Measured
    # 2026-09-15: identical to 4 decimal places on every moment. Drop it rather than carry it.
    if with_btc and sym != "BTC":
        btc = _from_close(rd.load_ohlcv("BTC")["close"], window)
        btc.columns = [f"btc_{c}" for c in btc.columns]
        out = out.join(btc, how="left")

    out = out.loc[(out.index >= lo) & (out.index <= hi)]
    return out.dropna()


def columns(pair: str, with_btc: bool = True) -> tuple[str, ...]:
    if with_btc and pair.split("/")[0] != "BTC":
        return FEATURES + tuple(f"btc_{c}" for c in FEATURES)
    return FEATURES
