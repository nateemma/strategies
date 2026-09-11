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


def directional_label(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON,
                      min_gain: float = PRODUCTION_THRESHOLD) -> LabelTarget:
    """The production gbb label: 1 where forward gain clears min_gain within the horizon.

    Delegates to the framework's own labeler so this feature trains on the SAME target the
    classifier did, rather than a re-implementation that could drift from it.
    """
    from TrainingSignals import labels_forward_return_mae_cap
    y = labels_forward_return_mae_cap(df, horizon=horizon, min_gain=min_gain).to_numpy()
    valid = np.ones(len(y), dtype=bool)
    valid[-horizon:] = False          # the last `horizon` bars cannot see their own future
    return LabelTarget("directional", horizon, min_gain,
                       np.nan_to_num(y, nan=0.0), valid)


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
