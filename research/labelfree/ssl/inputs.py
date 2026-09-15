"""The two things the encoder can consume (T004, T005).

Both are reported separately because they fail for DIFFERENT reasons and conflating them would
lose which:

  engineered  the 24-column production feature matrix. Cannot add information the indicators
              lack, so a null here says SSL finds no exploitable STRUCTURE the classifier misses.
  ohlcv       raw windows. The strongest form of the hypothesis -- and the input a learned 1D-CNN
              already failed on in this repo at rho 0.04-0.06.

RAW OHLCV IS NOT USED RAW. AGENT_GUIDE.md:546 prohibits raw price and raw volume as NN inputs
because their ranges differ by pair, which is the same pair-agnosticism rule this feature later
applies to the embedding's OUTPUT. Applying it to the encoder's INPUT as well is not a
convenience: an encoder fed raw levels would learn per-pair scale and the resulting embedding
could not pass FR-012a. So the window becomes log returns plus a volume ratio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree import features as FT                   # noqa: E402


def engineered(pair: str, start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    """(features, open price). The 24-column FeatureNormalizer.include_list matrix."""
    feat, px, _vq, _ = FT.build_features(pair, start, end)
    feat, px = np.asarray(feat, dtype=np.float32), np.asarray(px, dtype=np.float64)
    n = min(len(feat), len(px))
    return feat[:n], px[:n]


def ohlcv(pair: str, start: str, end: str, vol_window: int = 96
          ) -> tuple[np.ndarray, np.ndarray]:
    """(normalised OHLCV, open price) — log returns and a volume ratio, never raw levels."""
    from regime import regimedata as rd
    import pandas as pd

    df = rd.load_ohlcv(pair.split("/")[0])
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    df = df[(df.index >= lo) & (df.index <= hi)]
    if df.empty:
        return np.zeros((0, 5), np.float32), np.zeros(0)

    c = df["close"].to_numpy(float)
    prev = np.concatenate(([c[0]], c[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        cols = [
            np.log(df["open"].to_numpy(float) / prev),          # open  vs previous close
            np.log(df["high"].to_numpy(float) / prev),          # high  vs previous close
            np.log(df["low"].to_numpy(float) / prev),           # low   vs previous close
            np.log(c / prev),                                   # close vs previous close
            # LOG of the volume ratio, not the ratio itself. Measured on f07/AAVE the raw ratio
            # has sd 3.43 against the close-return channel's 0.0072 -- 477x -- with a max of
            # 73.7. A contrastive loss over channels that mismatched would learn almost entirely
            # from volume spikes. log1p brings it to the same order while staying a ratio, so it
            # remains pair-agnostic (AGENT_GUIDE.md:546).
            np.log1p(df["volume"].to_numpy(float)
                     / np.maximum(
                         df["volume"].rolling(vol_window, min_periods=5).mean().to_numpy(float),
                         1e-9)),
        ]
    X = np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32), df["open"].to_numpy(float)


def standardise(X: np.ndarray, stats: tuple | None = None):
    """Per-channel z-score. Returns (X_scaled, stats) so the SAME stats fit on the encoder's
    split can be applied downstream — fitting them per split would leak scale across the
    boundary, and fitting them per pair is what keeps the input pair-agnostic.
    """
    if stats is None:
        mu = X.mean(axis=0)
        sd = np.maximum(X.std(axis=0), 1e-8)
        stats = (mu, sd)
    mu, sd = stats
    return ((X - mu) / sd).astype(np.float32), stats


LOADERS = {"engineered": engineered, "ohlcv": ohlcv}


def forward_return(px: np.ndarray, horizon: int) -> np.ndarray:
    """Simple forward return over `horizon` bars, aligned to the bar it is known from."""
    n = len(px)
    if n <= horizon:
        return np.zeros(0)
    return px[horizon:] / px[:n - horizon] - 1.0
