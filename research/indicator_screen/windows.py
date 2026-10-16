"""037 windows (research R3, R15): the 15 fold test windows of plan_wf_aro_g24, the distant era E0,
purge, training-row masks and the late-window lock.

early = f00-f07 (2023-01 -> 2024-12), late = f08-f14 (2025-01 -> 2026-09), E0 = 2021-04-01 -> 2022-12-31.
Early and late are ADJACENT; E0 is the distant era Principle II needs (spec FR-018).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PLAN = STRAT / "evaluation" / "folds" / "plan_wf_aro_g24.json"
PREREG = HERE / "preregistration.json"

BAR = pd.Timedelta("15min")
WARMUP_BARS = 1000           # excluded from every score, per pair (R3)
LABEL_HORIZON = 48           # TrainingConfig.HORIZON: the gbb_path label look-ahead
DATA_START = "2021-01-01"
DATA_END = "2026-09-07"
E0 = ("2021-04-01", "2023-01-01")   # [start, end): 2021-04-01 -> 2022-12-31
EARLY = tuple(f"f{i:02d}" for i in range(8))
LATE = tuple(f"f{i:02d}" for i in range(8, 15))


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


@dataclass(frozen=True)
class Window:
    id: str
    role: str                     # "era" | "test"
    half: str                     # "e0" | "early" | "late"
    start: pd.Timestamp           # inclusive
    end: pd.Timestamp             # exclusive
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None   # = start for folds (anchored, train ends where test starts)

    def mask(self, dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        d = pd.DatetimeIndex(dates)
        return np.asarray((d >= self.start) & (d < self.end))


def folds(plan: Path = PLAN) -> list[Window]:
    p = json.loads(Path(plan).read_text())
    out = []
    for f in p["folds"]:
        tr0, tr1 = f["train_timerange"].split("-")
        te0, te1 = f["test_timerange"].split("-")
        out.append(Window(f["fold_id"], "test", "early" if f["fold_id"] in EARLY else "late",
                          _ts(te0), _ts(te1), _ts(tr0), _ts(tr1)))
    return out


def e0() -> Window:
    return Window("E0", "era", "e0", _ts(E0[0]), _ts(E0[1]), None, None)


def all_windows(halves=("e0", "early", "late")) -> list[Window]:
    return [w for w in [e0(), *folds()] if w.half in halves]


def by_id(wid: str) -> Window:
    return next(w for w in [e0(), *folds()] if w.id == wid)


def purge(target: str) -> int:
    """Bars between the last training row and the test start: the target's look-ahead."""
    if target.startswith("label:"):
        return LABEL_HORIZON
    return int(target.rsplit(":", 1)[1])      # regime ids end in ":<N bars>"


def train_rows_mask(dates, window: Window, target: str) -> np.ndarray:
    """Rows a model for `window` may train on: after the per-pair warm-up and before
    test_start - purge (in bars). `dates` is one pair's date column, in order."""
    if window.train_start is None:
        raise ValueError(f"{window.id} has no training span")
    d = pd.DatetimeIndex(dates)
    ok = np.asarray((d >= window.train_start) & (d < window.start - purge(target) * BAR))
    ok[:WARMUP_BARS] = False
    return ok


class LateLockError(RuntimeError):
    pass


def late_lock(path: Path = PREREG, repo: Path = STRAT) -> str:
    """Refuse unless the pre-registration is committed with no working-tree change (R15).
    Returns the sha of the last commit touching it."""
    path = Path(path)
    if not path.exists():
        raise LateLockError(f"{path.name} missing: late windows are locked")
    rel = str(path.resolve().relative_to(Path(repo).resolve()))
    tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel],
                             capture_output=True, text=True)
    if tracked.returncode != 0:
        raise LateLockError(f"{rel} is not committed: late windows are locked")
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", rel],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise LateLockError(f"{rel} has uncommitted changes: late windows are locked")
    return subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%h", "--", rel],
                          capture_output=True, text=True, check=True).stdout.strip()


def require_lock_for(windows: list[Window]) -> str | None:
    """Call before scoring: checks the lock when any late window is requested."""
    if any(w.half == "late" for w in windows):
        return late_lock()
    return None
