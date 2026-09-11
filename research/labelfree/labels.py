"""Forward-looking targets and their leakage controls (T003, T004, T005).

Every target here looks FORWARD, so every one carries a purge and an embargo. Brief 02's
arms all contain a forward-looking term — Arm A's warm-start classifier is label-trained,
Arm B has a cross-entropy term, Arm C predicts forward volatility — so none of them is
exempt (FR-013).

Purge length is the LABEL HORIZON, not the peak window. Measured from the framework:
`TrainingConfig.HORIZON = 48` and the peak filter was fixed to use HORIZON rather than
PEAK_WINDOW (commit 8d42c01), so a label at bar t can see up to t+48. Purging the shorter
PEAK_WINDOW (6) would leave 42 bars of leakage at every fold boundary.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Framework"))

DEFAULT_HORIZON = 48          # TrainingConfig.HORIZON — the label's forward reach
PRODUCTION_THRESHOLD = 0.007  # MIN_BUY_GAIN_THRESHOLD, production
ALTERNATIVE_THRESHOLD = 0.02  # the value swept in commit 7ddcfb2


@dataclass(frozen=True)
class LabelTarget:
    """A forward-looking target plus the mask saying where it is usable."""
    kind: str                  # "directional" | "forward_realised_vol"
    horizon_bars: int
    threshold: float | None
    values: np.ndarray         # (n_pairs, T) or (T,)
    valid_mask: np.ndarray     # False where the target cannot be computed causally

    def describe(self) -> dict:
        return {"kind": self.kind, "horizon_bars": self.horizon_bars,
                "threshold": self.threshold,
                "valid_share": float(self.valid_mask.mean())}


# TradingAction, mirrored from Framework/BaseStrategy so this module does not import a
# strategy base just for three integers.
SELL, HOLD, BUY = 0, 1, 2


def directional_label(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON,
                      min_gain: float = PRODUCTION_THRESHOLD) -> LabelTarget:
    """The production 3-class gbb target: SELL=0, HOLD=1, BUY=2.

    Uses `labels_gbb_path` / `labels_gbb_path_sell`, NOT plain gbb. `TrainingConfig
    .TRAINING_TYPE = 27` (gbb_path), changed from 17 (gbb) on 2026-09-08, and NNNC_MLX does
    not override it. Warm-starting Arm A from a classifier trained on a DIFFERENT label
    than production would make the arm measure the wrong prior entirely.

    (Two commits look contradictory here: f26e45b rejected gbb_path as a strategy change,
    then e7cf8cf adopted it as the framework LABEL default on better-specified-target
    grounds — ceilings +15.6pp on NNNC guards-off. It is the label that matters for this
    feature.)

    The path-aware pair drops buy labels whose stop would fire before the target, and sell
    labels the take-profit would pre-empt, so the label certifies a move the strategy can
    actually harvest. `min_gain` is the parameter swept 0.007 -> 0.02 in commit 7ddcfb2.

    REQUIRES an indicator-populated dataframe: the gbb labelers gate on guard_metric,
    bb_width and rvol. Passing raw OHLCV silently yields an all-HOLD target.

    Conflict resolution follows the framework: sells override buys. The two labelers were
    deliberately made disjoint at guard_threshold 0.3 (2026-09-08), so overlap should be
    rare; where it remains, matching the framework's precedence keeps this target identical
    to the classifier's.
    """
    from TrainingSignals import labels_gbb_path, labels_gbb_path_sell
    buy = labels_gbb_path(df, min_gain=min_gain, horizon=horizon).to_numpy()
    sell = labels_gbb_path_sell(df, horizon=horizon).to_numpy()

    y = np.full(len(buy), HOLD, dtype=np.int64)
    y[np.nan_to_num(buy, nan=0.0) > 0.5] = BUY
    y[np.nan_to_num(sell, nan=0.0) > 0.5] = SELL      # sells override buys

    required = ("guard_metric", "bb_width", "rvol")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"labels_gbb_path gates on {required}; missing {missing}. Pass an "
            "indicator-populated dataframe — raw OHLCV silently yields an all-HOLD target.")

    valid = np.ones(len(y), dtype=bool)
    valid[-horizon:] = False          # the last `horizon` bars cannot see their own future
    return LabelTarget("directional", horizon, min_gain, y, valid)


def one_hot(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """3-class one-hot, as MLXClassifierNary.train() expects for `train_results`."""
    return np.eye(n_classes, dtype=np.float32)[y.astype(int)]


def forward_realised_vol(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> LabelTarget:
    """Forward realised volatility — a MAGNITUDE, carrying no directional information.

    Arm C's auxiliary target. It exists to separate "labels help because they carry
    direction" from "any auxiliary supervision regularises the objective", so it must be
    sign-free by construction: the standard deviation of forward log returns is
    non-negative for every bar, which the test asserts rather than assumes.
    """
    close = df["close"].to_numpy(dtype=np.float64)
    r = np.diff(np.log(np.maximum(close, 1e-12)), prepend=np.log(max(close[0], 1e-12)))
    n = len(r)
    out = np.full(n, np.nan)
    # Forward window [t+1, t+horizon]
    for t in range(n - horizon):
        out[t] = r[t + 1: t + 1 + horizon].std()
    valid = ~np.isnan(out)
    return LabelTarget("forward_realised_vol", horizon, None,
                       np.nan_to_num(out, nan=0.0), valid)


def purge_embargo_mask(n_bars: int, boundaries: list[int],
                       horizon: int = DEFAULT_HORIZON) -> np.ndarray:
    """False for bars whose target would leak across a fold boundary (FR-013).

    At a boundary b: the `horizon` bars BEFORE it are purged, because their forward window
    reaches past b into the held-out range. The `horizon` bars AFTER it are embargoed,
    because they overlap the training range's own forward windows (Lopez de Prado ch. 7).
    """
    ok = np.ones(n_bars, dtype=bool)
    for b in boundaries:
        ok[max(0, b - horizon): min(n_bars, b + horizon + 1)] = False
    return ok


def apply_leakage_control(target: LabelTarget, boundaries: list[int],
                          horizon: int | None = None) -> tuple[np.ndarray, dict]:
    """Combine a target's own validity with purge/embargo. Returns (mask, record).

    The record carries the values ACTUALLY applied, not the values requested, because a
    run whose purge silently fell to zero must be void rather than caveated (FR-014).
    """
    h = horizon if horizon is not None else target.horizon_bars
    n = target.values.shape[-1]
    pe = purge_embargo_mask(n, boundaries, h)
    mask = target.valid_mask & pe
    return mask, {
        "purge_bars": h, "embargo_bars": h, "boundaries": list(boundaries),
        "n_bars": int(n), "usable_share": float(mask.mean()),
        "target": target.describe(),
    }
