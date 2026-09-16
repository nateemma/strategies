"""Principle IX -- is this gate the incumbent's rvol guard in disguise? (T029)

RUNS BEFORE ANY PERFORMANCE NUMBER. The incumbent already gates entries on relative volume
>= 2.0, and its guard chain rejects roughly 97% of candidates with rvol dominant. A model
fitted on realised and downside volatility is at serious risk of being that same axis wearing a
different name, and Principle IX requires correlation against what is already there to be
checked FIRST.

Both outcomes are pre-registered and both are informative:

  novel_block_rate <  0.20  -> the gate is inert BECAUSE RVOL GOT THERE FIRST. A defect report
                              under Principle IV, and specifically NOT "regime gating does not
                              help".
  novel_block_rate >= 0.20  -> the blocked set is distinct and the comparison is meaningful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import NOVEL_BLOCK_FLOOR
from . import features as FE

RVOL_GUARD = 2.0          # the incumbent's production entry gate


def rvol_series(pair: str, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
    """volume / rolling(20).mean() -- the quantity the STOP and the entry guard use.

    Deliberately not the `rvol` COLUMN, which is an ewm(span=10): same name, different
    quantity (BaseStrategy.py:1043-1048). Getting that wrong would compare the jump gate
    against something the incumbent does not actually do.
    """
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from regime import regimedata as rd

    v = rd.load_ohlcv(pair.split("/")[0])["volume"]
    r = v / v.rolling(20, min_periods=5).mean()
    return r.loc[(r.index >= lo) & (r.index <= hi)].fillna(0.0)


def overlap(states: pd.DataFrame, rvol: pd.Series, high_vol_state: int) -> dict:
    """Contingency of jump-gate blocks against bars rvol already rejects."""
    s = states.set_index("timestamp")["state"]
    idx = s.index.intersection(rvol.index)
    s, r = s.loc[idx], rvol.loc[idx]

    jump_blocks = (s == high_vol_state).to_numpy()
    rvol_blocks = (r < RVOL_GUARD).to_numpy()          # the incumbent admits only rvol >= 2.0

    n_blocked = int(jump_blocks.sum())
    n_both = int((jump_blocks & rvol_blocks).sum())
    novel = int((jump_blocks & ~rvol_blocks).sum())
    rate = (novel / n_blocked) if n_blocked else 0.0
    return {
        "n_bars": int(len(idx)),
        "n_blocked": n_blocked,
        "n_rvol_blocked": int(rvol_blocks.sum()),
        "n_both": n_both,
        "n_novel": novel,
        "novel_block_rate": float(rate),
        "verdict": "redundant_with_rvol" if rate < NOVEL_BLOCK_FLOOR else "distinct",
    }
