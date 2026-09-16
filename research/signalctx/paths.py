"""Per-bar unrealised P&L for an open trade (T007, T008).

WHY THIS MODULE EXISTS. The confound test asks what a position was worth AT THE MOMENT a repeat
signal arrived. Backtest archives record open and close rates and dates, never the path between,
so the path must be rebuilt.

ENTRY PRICES COME FROM THE ARCHIVES, NOT THE POOLED RECORD. `evaluation/pooled_nnnc.json` carries
only dates, `profit_ratio` and `profit_abs` -- no `open_rate`. The per-fold archives carry
`open_rate`, `close_rate`, `min_rate` and `max_rate`, so they are the source. All 15 control
folds have one (verified 2026-09-16; 11 unrelated archives in that directory are CRC-corrupt and
are skipped).

ADDITIVE, NEVER COMPOUNDED. Marks convert to percentage points at a fixed 9.0 pp per unit
`profit_ratio` (900 stake on a 10,000 wallet). Compounding pooled trades once read a real
+23.46 pp/yr as -19.06 (brief 013).

WHAT A close-BASED MARK CANNOT SEE. Stop and take-profit exits fill INTRABAR, so the final
close-based mark does not equal the recorded `profit_ratio` for those trades -- that is a
property of the exit, not an error in the path. `reconcile()` therefore validates the
CONSTRUCTION against `close_rate`, and any claim about stop proximity must use `min_rate`/`low`
and say so.
"""
from __future__ import annotations

import glob
import json
import os
import re
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd

from . import RECONCILE_TOL, ROOT, STAKE_PP_PER_RATIO

ARCHIVES = ROOT.parent.parent / "user_data/backtest_results"


@lru_cache(maxsize=None)
def _archive_index() -> dict:
    """fold_id -> trade list, from the most recent readable archive per control class."""
    out = {}
    for z in sorted(glob.glob(str(ARCHIVES / "*.zip")), key=os.path.getmtime):
        try:
            with zipfile.ZipFile(z) as zf:
                name = [x for x in zf.namelist() if x.endswith(".json") and "config" not in x][0]
                d = json.loads(zf.read(name))
        except Exception:
            continue                      # CRC-corrupt archives are skipped, not fatal
        st = list(d["strategy"].keys())[0]
        m = re.fullmatch(r"NNNC_WF_(f\d+)", st)
        if m:
            out[m.group(1)] = d["strategy"][st]["trades"]
    return out


def trades(fold_id: str) -> pd.DataFrame:
    tr = _archive_index().get(fold_id)
    if tr is None:
        raise SystemExit(f"no readable control archive for {fold_id}")
    df = pd.DataFrame(tr)
    df["open_date"] = pd.to_datetime(df["open_date"], utc=True)
    df["close_date"] = pd.to_datetime(df["close_date"], utc=True)
    return df


def unrealised(trade: dict, panel: dict, mark: str = "close") -> pd.Series:
    """Per-bar unrealised P&L in percentage points, from entry to exit inclusive."""
    idx = pd.DatetimeIndex(panel["index"])
    lo = pd.Timestamp(trade["open_date"]); hi = pd.Timestamp(trade["close_date"])
    sel = (idx >= lo) & (idx <= hi)
    if not sel.any():
        return pd.Series(dtype=float)
    px = np.asarray(panel[mark], dtype=float)[sel]
    ratio = px / float(trade["open_rate"]) - 1.0
    return pd.Series(ratio * STAKE_PP_PER_RATIO, index=idx[sel], name="unrealised_pp")


def reconcile(trade: dict) -> float:
    """|NET close-rate return - recorded profit_ratio|, in profit_ratio units.

    NET, not gross. A first version compared the gross ratio and every one of 378 trades missed
    by 0.00202 +/- 0.0003 -- a residual so tight across all 15 folds that it was obviously a
    constant, not an error: the round-trip fee at freqtrade's 0.1% per side. The fix is the
    formula, not the tolerance.

    Fees are taken from the trade's own `fee_open`/`fee_close` rather than assumed, because a
    hard-coded 0.001 would silently drift if the config's fee ever changed.
    """
    fo = float(trade.get("fee_open", 0.0) or 0.0)
    fc = float(trade.get("fee_close", 0.0) or 0.0)
    net = (float(trade["close_rate"]) * (1.0 - fc)) / (float(trade["open_rate"]) * (1.0 + fo)) - 1.0
    return abs(net - float(trade["profit_ratio"]))


def reconcile_fold(fold_id: str) -> dict:
    df = trades(fold_id)
    res = np.array([reconcile(t) for _, t in df.iterrows()])
    return {"fold_id": fold_id, "n": int(len(res)), "max": float(res.max()),
            "mean": float(res.mean()), "within_tol": int((res <= RECONCILE_TOL).sum()),
            "tol": RECONCILE_TOL, "ok": bool((res <= RECONCILE_TOL).all())}
