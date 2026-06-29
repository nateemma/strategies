"""Causal, lookahead-free Fibonacci-retracement features (candidate indicator).

Status: EXPERIMENTAL — signal-check with DebugSessionFeatures before adopting.
If it earns its place, the home is utils/DataframePopulator (computation) +
include_list, not here.

Both variants are PAIR-AGNOSTIC: outputs are normalized to the swing range
(position in [0,1], distances in range-units), so no raw price/volume scale
leaks into the model (the no-pair-specific-data rule).

  add_fib_features(df, window)       FULLY VECTORIZED, no loop. Trailing rolling
                                     swing hi/lo -> fib_position + fib_nearest.

  add_fibodynamic_features(df, ...)  Faithful NinjaTrader "Fibodynamic" 50%-reset.
                                     The reset is path-dependent, so it needs ONE
                                     causal sequential pass — it canNOT be fully
                                     vectorized without lookahead, and lookahead
                                     is the thing we refuse.

Lookahead safety: every value at bar t uses only bars <= t. Rolling windows are
trailing (pandas default; NO center=True); the sequential pass reads only
indices <= i. The module's __main__ smoke test proves this empirically via
truncation-invariance (prefix of f(x) == f(x[:k])). Still run
`freqtrade lookahead-analysis` on any strategy that uses these before trusting a
backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Standard retracement lines used as soft S/R reference points.
_FIB_LEVELS = np.array([0.0, 0.382, 0.5, 0.618, 1.0])


def _nearest_level_dist(pos: np.ndarray) -> np.ndarray:
    """Distance from each position to the nearest fib line (nonlinear — encodes
    'price is sitting on a retracement S/R level', which fib_position alone,
    being affine in the levels, does not)."""
    return np.min(np.abs(pos[:, None] - _FIB_LEVELS[None, :]), axis=1)


def add_fib_features(df: pd.DataFrame, window: int = 96, prefix: str = "fib") -> pd.DataFrame:
    """Vectorized causal retracement position over a fixed trailing window."""
    swing_hi = df["high"].rolling(window, min_periods=2).max()   # trailing => causal
    swing_lo = df["low"].rolling(window, min_periods=2).min()
    rng = swing_hi - swing_lo
    pos = ((df["close"] - swing_lo) / rng.where(rng > 0)).clip(0.0, 1.0).fillna(0.5)
    df[f"{prefix}_position"] = pos.astype("float32")
    df[f"{prefix}_nearest"] = pd.Series(
        _nearest_level_dist(pos.to_numpy()), index=df.index
    ).astype("float32")
    return df


def add_fibodynamic_features(
    df: pd.DataFrame, max_window: int = 192, prefix: str = "fibdyn"
) -> pd.DataFrame:
    """Faithful 50%-reset Fibodynamic. The active swing range grows with each
    bar; when close crosses the 50% line (or the leg exceeds max_window — the
    adaptive cap), the range resets to the current bar and a new leg begins.
    Single causal pass: the loop reads only bars <= i."""
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    n = len(df)

    pos = np.full(n, 0.5)
    age = np.zeros(n)
    cur_hi, cur_lo, prev_p, leg = -np.inf, np.inf, 0.5, 0
    for i in range(n):                       # causal: only indices <= i are read
        if high[i] > cur_hi:
            cur_hi = high[i]
        if low[i] < cur_lo:
            cur_lo = low[i]
        leg += 1
        rng = cur_hi - cur_lo
        p = 0.5 if rng <= 0 else (close[i] - cur_lo) / rng
        p = 0.0 if p < 0.0 else 1.0 if p > 1.0 else p
        pos[i] = p
        age[i] = leg
        crossed_mid = (prev_p - 0.5) * (p - 0.5) < 0.0
        if crossed_mid or leg >= max_window:
            cur_hi, cur_lo, leg = high[i], low[i], 0   # reset takes effect next bar
        prev_p = p

    df[f"{prefix}_position"] = pos.astype("float32")
    df[f"{prefix}_age"] = (np.minimum(age, max_window) / max_window).astype("float32")
    df[f"{prefix}_nearest"] = _nearest_level_dist(pos).astype("float32")
    return df


if __name__ == "__main__":
    # Smoke + empirical lookahead proof on one pair.
    import sys
    from pathlib import Path

    strat = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(strat))
    sys.path.insert(0, str(strat / "Debug"))
    import DebugAnalyseIndicators as D  # noqa: E402

    data_dir = strat.parent / "data" / "binanceus"
    df = D.load_pair_data(data_dir, "ZEC_USDT", "15m")

    full = add_fibodynamic_features(add_fib_features(df.copy()))
    cols = [c for c in full.columns if c.startswith(("fib_", "fibdyn_"))]
    print("columns:", cols)
    print(full[cols].describe().loc[["min", "max", "mean"]].T)

    # Truncation-invariance: f(x[:k]) must equal f(x)[:k] exactly for a causal
    # transform. Any mismatch => lookahead.
    k = len(df) // 2
    trunc = add_fibodynamic_features(add_fib_features(df.iloc[:k].copy()))
    ok = all(
        np.allclose(full[c].to_numpy()[:k], trunc[c].to_numpy(), equal_nan=True)
        for c in cols
    )
    print(f"\nLOOKAHEAD-FREE (truncation-invariant over {k} bars): {ok}")
