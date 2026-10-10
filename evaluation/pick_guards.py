"""Per-fold, leakage-free GUARD selection for feature 022.

SIBLING OF `pick_threshold.py`, deliberately. That module selects a per-fold
`prediction_threshold` on the fold's own training window and routes the choice into the run
through a plan key; this one does the same for guard thresholds. It REUSES
`pick_threshold.TrainWindow`, whose `from_fold` asserts the window ends at or before the fold's
test window starts, rather than reimplementing that guarantee.

NOT A BACKTEST, and that is what keeps the Principle VII gate affordable. Scoring 64
configurations across 15 folds as real backtests is ~960 runs and roughly 30 hours, which would
make the gate more expensive than the arm it exists to gate. Instead a configuration is scored
from the per-bar OUTCOME TABLE (`research/outcomes/table.py`), keyed by `exit_config_hash` so a
changed exit invalidates it: admitted-bar mask -> concurrency selection -> pooled mean.

THE SCORE IS A PROXY AND IS NAMED ONE. `proxy_score`, never `score`, and never anything
containing "return" (FR-018). Feature 021's central error was a table-derived figure read as a
realisable return; the same instrument there matched live mean profit_ratio to 3.6e-5 on one
population and had the WRONG SIGN on another. Its fidelity is a measured property, which is why
`--audit-proxy` exists and must pass before the gate is licensed.

THE GUARD CONDITIONS ARE COPIED FROM THE EXECUTABLE PATH, not from a comment. Read off
`Framework/BaseStrategy.populate_entry_trend`:

    MANDATORY, outside the entry_enable_guards flag (commit e7cf8cf, 2026-09-08):
        volume > 0.0
        rvol    > entry_rvol_threshold
    QUALITY, inside the flag:
        atr_pct_roll > entry_atr_pct
        guard_metric < entry_guard_threshold
        close_norm   < entry_close_norm_threshold
        adx          > entry_adx_threshold
        bb_width     > entry_bb_width_threshold

BaseStrategy's `buy_params` comment still claims the flag also gates `volume > 0`. It does not,
and reading that comment instead of this code already produced one wrong explanation in feature
021. Only the three tuned axes vary here; the other quality thresholds stay at their open values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

GRID_FILE = Path(__file__).resolve().parent / "folds" / "guard_grid_022.json"

#: FR-012: a configuration admitting fewer than this many trades PER 90-DAY-EQUIVALENT is
#: rejected. Derived from the harness's own power record -- 378 pooled trades give the
#: 3.350 pp/yr floor, and 378/15 ~ 25 on a 90-day test window.
MIN_TRADES_PER_FOLD = 25

#: The window the floor above was derived for. Everything else scales against it.
FLOOR_REFERENCE_DAYS = 90


def floor_for_window(window_days: float) -> int:
    """The trade floor for a window of this length, RATE-MATCHED to the 90-day reference.

    DEFECT THIS FIXES, found 2026-10-10 by the user asking why the adaptive scheme never fired.
    The floor was applied as an ABSOLUTE count to both windows, but this harness's train
    windows are ANCHORED and grow: 730 days at f00 to 1990 at f14, against 90-day test
    windows -- 8.1x to 23.4x longer, 15.2x on average. So a flat 25 on the train window
    admitted configurations trading ~1.7 trades per 90 days, which could never clear 25 on
    test. The adaptive scheme was selecting unmeasurable configurations because nothing
    stopped it: the constraint was wrong, not the adaptation.

    Because the ratio VARIES across folds (8.1x to 23.4x), no single absolute number works for
    both windows -- it has to be a rate.
    """
    return int(round(MIN_TRADES_PER_FOLD * float(window_days) / FLOOR_REFERENCE_DAYS))

#: Open values for the quality thresholds this feature does NOT tune, taken from
#: BaseStrategy.buy_params so a scored configuration differs from arm A in the three axes only.
UNTUNED_OPEN = {
    "entry_guard_threshold": 1.1,
    "entry_close_norm_threshold": 1.1,
    "entry_bb_width_threshold": -1.0,
}


def load_grid() -> list[dict]:
    """The 64 pinned configurations (FR-005). Pinned before any scoring, so the candidate set
    is finite and inspectable rather than chosen during the search."""
    return json.loads(GRID_FILE.read_text())["configs"]


def admitted_mask(features: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Bars this guard configuration admits, reproducing the live condition order.

    `entry_enable_guards` is assumed True -- it is what applies the quality thresholds at all,
    and FR-008 requires it in every arm. The trigger itself is NOT applied here; the caller
    conjoins it, so this function is purely the guard chain.
    """
    n = len(features)

    def col(name: str, fill: float) -> np.ndarray:
        if name not in features.columns:
            return np.full(n, fill, dtype=float)
        return np.nan_to_num(np.asarray(features[name], dtype=float), nan=fill)

    # MANDATORY -- outside the flag in the live path
    m = col("volume", 0.0) > 0.0
    m &= col("rvol", 0.0) > float(cfg["entry_rvol_threshold"])
    # QUALITY -- inside the flag. NaN->fill chosen so a missing column cannot silently admit
    # a bar: atr_pct_roll/adx/bb_width fill LOW (fail a `>`), guard_metric/close_norm fill HIGH
    # (fail a `<`).
    m &= col("atr_pct_roll", -np.inf) > float(cfg["entry_atr_pct"])
    m &= col("adx", -np.inf) > float(cfg["entry_adx_threshold"])
    m &= col("guard_metric", np.inf) < float(
        cfg.get("entry_guard_threshold", UNTUNED_OPEN["entry_guard_threshold"]))
    m &= col("close_norm", np.inf) < float(
        cfg.get("entry_close_norm_threshold", UNTUNED_OPEN["entry_close_norm_threshold"]))
    m &= col("bb_width", -np.inf) > float(
        cfg.get("entry_bb_width_threshold", UNTUNED_OPEN["entry_bb_width_threshold"]))
    return m


# =================================================================================================
# Scoring one configuration over one window
# =================================================================================================

def score_config(frames, trigger: str, cfg: dict, start, end, max_open: int = 10) -> dict:
    """`GridScore` for one configuration over one date window.

    `frames` is [(features, outcomes)] per pair. The window is applied to BOTH frames by date,
    so a configuration is only ever scored on bars inside it -- that is the leakage boundary
    when `start`/`end` come from a `TrainWindow`.
    """
    from research.outcomes import labels as _labels
    from research.outcomes.leakage import OUTCOME_COLUMN_PROFIT
    from research.outcomes import leaderboard as _lb

    sel = []
    for f, o in frames:
        d = pd.to_datetime(f["date"], utc=True)
        win = (d >= pd.Timestamp(start, tz="UTC")) & (d < pd.Timestamp(end, tz="UTC"))
        if not win.any():
            continue
        fw, ow = f.loc[win].reset_index(drop=True), o.loc[win].reset_index(drop=True)
        ent = _labels.entry_mask(fw, ow, trigger).to_numpy() & admitted_mask(fw, cfg)
        o2 = ow.copy()
        o2[OUTCOME_COLUMN_PROFIT] = np.where(ent, ow[OUTCOME_COLUMN_PROFIT].to_numpy(), np.nan)
        sel.append((fw, o2, ent))

    if not sel:
        return {"proxy_score": float("nan"), "n_admitted": 0, "n_taken": 0, "below_floor": True}

    # concurrency: merge all pairs' admitted bars into one time-ordered stream under max_open
    ents, profs, holds, dates = [], [], [], []
    from research.outcomes.leakage import OUTCOME_COLUMN_HOLD_BARS
    for fw, o2, ent in sel:
        idx = np.flatnonzero(ent)
        if idx.size == 0:
            continue
        dn = pd.to_datetime(fw["date"], utc=True).to_numpy().astype("datetime64[ns]").astype(np.int64)
        h = np.nan_to_num(np.asarray(o2[OUTCOME_COLUMN_HOLD_BARS], dtype=float), nan=0.0)
        step = np.median(np.diff(dn)) if dn.size > 1 else 15 * 60 * 1_000_000_000
        dates.append(dn[idx]); profs.append(np.asarray(o2[OUTCOME_COLUMN_PROFIT], float)[idx])
        holds.append(dn[idx] + (h[idx] * step).astype(np.int64))
    n_adm = int(sum(len(x) for x in dates))
    if n_adm == 0:
        return {"proxy_score": float("nan"), "n_admitted": 0, "n_taken": 0, "below_floor": True}
    entry = np.concatenate(dates); prof = np.concatenate(profs); exit_ = np.concatenate(holds)
    order = np.argsort(entry, kind="stable")
    entry, prof, exit_ = entry[order], prof[order], exit_[order]
    open_until, taken = [], []
    for i in range(entry.size):
        t = entry[i]
        open_until = [x for x in open_until if x > t]
        if len(open_until) < max_open:
            taken.append(i); open_until.append(exit_[i])
    p = prof[np.asarray(taken, dtype=int)] if taken else np.empty(0)
    p = p[np.isfinite(p)]
    return {"proxy_score": float(p.mean()) if p.size else float("nan"),
            "n_admitted": n_adm, "n_taken": int(p.size),
            "below_floor": bool(p.size < MIN_TRADES_PER_FOLD)}
