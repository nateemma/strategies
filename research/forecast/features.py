"""038 learned-tier inputs (research R7): all trailing, computed per pair on the gap-filled frame.

- L: log returns over 1, 2, 4, 8, 16, 32, 64, 96 candles; realised volatility of 1-candle returns over
  16 and 96; log volume ratio (mean volume 16 / 96);
- LA: L + the 24 ARO model inputs (037 cache `base:` columns aliased by `aro:`);
- LW: L + on the trailing 64 one-candle log returns: SWT bior3.9 level-2 coefficients at the last
  position (cA2, cD2, cD1) and FFT magnitudes of bins 1-8 (only inputs passing truncation.py).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pywt
from numpy.lib.stride_tricks import sliding_window_view

from research.forecast import common as C

LAGS = (1, 2, 4, 8, 16, 32, 64, 96)
WAVE_WINDOW = 64
WAVELET = "bior3.9"
LW_COLS = ["swt_cA2", "swt_cD2", "swt_cD1", *[f"fft_{k}" for k in range(1, 9)]]


def aro_columns() -> list[str]:
    from research.indicator_screen import harvest as H

    m = json.loads(H.MANIFEST.read_text())
    return sorted({v for k, v in m["aliases"].items() if k.startswith("aro:")})


def set_L(fr: pd.DataFrame) -> pd.DataFrame:
    lc = fr["lc"]
    r1 = lc.diff()
    out = {f"ret_{k}": lc - lc.shift(k) for k in LAGS}
    out["vol_16"] = r1.rolling(16, min_periods=16).std()
    out["vol_96"] = r1.rolling(96, min_periods=96).std()
    v = fr["volume"].astype(float)
    out["vratio_16_96"] = np.log((v.rolling(16, min_periods=16).mean() + 1e-12)
                                 / (v.rolling(96, min_periods=96).mean() + 1e-12))
    return pd.DataFrame(out, index=fr.index).astype(np.float32)


def set_A(pair: str, fr: pd.DataFrame) -> pd.DataFrame:
    from research.indicator_screen import harvest as H

    cols = aro_columns()
    c = H.read_cache(pair, cols)
    c = c.set_index(pd.to_datetime(c["date"]))[cols]
    return c.reindex(pd.DatetimeIndex(fr["date"])).set_axis(fr.index).astype(np.float32)


def set_W(fr: pd.DataFrame, chunk: int = 20000) -> pd.DataFrame:
    """Decomposition inputs at each candle t from 1-candle log returns t-63..t (NaN before)."""
    r1 = fr["lc"].diff().to_numpy(float)
    n = len(r1)
    out = np.full((n, len(LW_COLS)), np.nan, dtype=np.float32)
    first = WAVE_WINDOW  # r1[0] is NaN, so the first full window ends at row 64
    if n > first:
        win = sliding_window_view(r1[1:], WAVE_WINDOW)  # window i covers r1[1+i .. 64+i] -> ends at row 64+i
        for s in range(0, len(win), chunk):
            w = win[s: s + chunk]
            cA2, cD2, cD1 = pywt.swt(w, WAVELET, level=2, trim_approx=True, axis=-1)
            mags = np.abs(np.fft.rfft(w, axis=-1))[:, 1:9]
            out[first + s: first + s + len(w)] = np.column_stack([cA2[:, -1], cD2[:, -1], cD1[:, -1], mags])
    return pd.DataFrame(out, index=fr.index, columns=LW_COLS)


ATROUS_LEVELS = 5  # scales 1-2, 2-4, 4-8, 8-16, 16-32 candles; smooth c5 ~ a 32-candle average


def atrous(lc: pd.Series, levels: int = ATROUS_LEVELS) -> tuple[list[pd.Series], pd.Series]:
    """Causal Haar a trous transform (Renaud, Starck & Murtagh 2005): c_j(t) = (c_{j-1}(t) +
    c_{j-1}(t - 2^(j-1))) / 2, w_j = c_{j-1} - c_j. One-sided, so every coefficient at t uses data
    <= t only (no padding, no wrap-around), and lc = sum(w_j) + c_J exactly."""
    c, details = lc.astype(float), []
    for j in range(1, levels + 1):
        c_new = 0.5 * (c + c.shift(2 ** (j - 1)))
        details.append(c - c_new)
        c = c_new
    return details, c


def set_T(fr: pd.DataFrame, levels: int = ATROUS_LEVELS) -> pd.DataFrame:
    """Multiscale AR inputs: each scale's coefficient now and one scale-step back, w_j(t) and
    w_j(t - 2^j); the smooth's slope c_J(t) - c_J(t - 2^J) and the price's distance from it."""
    w, cJ = atrous(fr["lc"], levels)
    out = {}
    for j, wj in enumerate(w, start=1):
        out[f"at_w{j}"] = wj
        out[f"at_w{j}_lag"] = wj.shift(2 ** j)
    out["at_smooth_slope"] = cJ - cJ.shift(2 ** levels)
    out["at_dist"] = fr["lc"] - cJ
    return pd.DataFrame(out, index=fr.index).astype(np.float32)


AT_COLS = [*[c for j in range(1, ATROUS_LEVELS + 1) for c in (f"at_w{j}", f"at_w{j}_lag")],
           "at_smooth_slope", "at_dist"]


def build(pair: str, fr: pd.DataFrame, fset: str, lw_cols: list[str] | None = None) -> pd.DataFrame:
    base = set_L(fr)
    if fset in ("L", "null"):
        return base
    if fset == "AT":
        return set_T(fr)
    if fset == "LA":
        return pd.concat([base, set_A(pair, fr)], axis=1)
    if fset == "LW":
        w = set_W(fr)
        return pd.concat([base, w[lw_cols if lw_cols is not None else LW_COLS]], axis=1)
    if fset == "LAT":
        return pd.concat([base, set_T(fr)], axis=1)
    raise ValueError(fset)
