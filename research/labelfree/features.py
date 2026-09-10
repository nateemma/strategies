"""Feature ingestion and pair-agnostic validation (T008a, T008b).

FR-001  per-pair feature tensors built by the EXISTING pipeline and scalers, so inputs are
        identical to what the control sees
FR-002  pair-agnostic features only: no raw price, no raw volume, nothing proportional to
        price (AGENT_GUIDE "Feature Engineering Rules"; constitution Principle X)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "Framework"))

from regime import regimedata as rd            # noqa: E402

SAVED = ROOT / "saved_data"

# Columns that are price- or volume-proportional and therefore forbidden as features.
FORBIDDEN_EXACT = {"open", "high", "low", "close", "volume", "date"}
FORBIDDEN_SUBSTR = ("_price", "sma", "ema", "vwap", "bb_upper", "bb_lower", "bb_middle",
                    "tema", "kama", "wma", "dema", "trima")


class FeatureValidationError(ValueError):
    pass


def validate_include_list(cols: list[str]) -> None:
    """Reject any raw-price / raw-volume / price-proportional column (FR-002).

    Raises rather than warns: a price-proportional feature is not a degraded model, it is
    a pair-specific one, which Principle X prohibits outright.
    """
    bad = []
    for c in cols:
        lc = c.lower()
        if lc in FORBIDDEN_EXACT:
            bad.append((c, "raw price/volume column"))
        elif any(s in lc for s in FORBIDDEN_SUBSTR) and not lc.endswith(("_pct", "_z", "_ratio", "_norm")):
            bad.append((c, "price-proportional indicator (needs _pct/_z/_ratio/_norm form)"))
    if bad:
        raise FeatureValidationError(
            "pair-agnostic rule violated (FR-002):\n" +
            "\n".join(f"  {c}: {why}" for c, why in bad))


def load_panel(pairs: list[str], window: str | None = None,
               start: str | None = None, end: str | None = None) -> dict[str, pd.DataFrame]:
    """Raw OHLCV per pair, restricted to a pinned window or an explicit range."""
    out = {}
    for p in pairs:
        df = rd.load_ohlcv(p.split("/")[0])
        if window:
            df = rd.window_slice(df, window)
        else:
            if start:
                df = df[df.index >= pd.Timestamp(start, tz="UTC")]
            if end:
                df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        out[p] = df
    return out


def sequence_tensor(feat: pd.DataFrame, seq_len: int = 16) -> np.ndarray:
    """(n_seq, seq_len, n_features) sliding windows, matching BaseNNStrategy.seq_len.

    Window i ends at bar i+seq_len-1, so the tensor row for a decision at bar t uses bars
    <= t only. Nothing here may look forward (FR-003).
    """
    a = feat.to_numpy(dtype=np.float32)
    n, k = a.shape
    if n < seq_len:
        return np.empty((0, seq_len, k), dtype=np.float32)
    idx = np.arange(seq_len)[None, :] + np.arange(n - seq_len + 1)[:, None]
    return a[idx]


def scaler_available() -> dict[str, bool]:
    return {n: (SAVED / n).exists() for n in ("main_scaler.pkl", "main_tensor_scaler.pkl")}
