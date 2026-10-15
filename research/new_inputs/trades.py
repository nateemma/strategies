"""035 T013a: the early width-book trade set every 035 stage scores (FR-008; research R8).

034's S1 instrument, unchanged: the width book (gbb rule + bb_width >= the previous fold's
count-matched cut, width_twin.json) simulated once per early fold from the control's ARO tables
with the 24 h grace. Each trade carries its signal date (entry bar - 1), bb_width and the
control's p_buy there, and the control's 0.5-share agree. `trades()` refuses to return unless the
trade count (412) and the control-at-share gap (0.1004 +- 0.002) reproduce, and never returns a
late fold.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

EARLY = tuple(f"f{i:02d}" for i in range(8))
EXPECTED_TRADES = 412
EXPECTED_GAP, GAP_TOL = 0.1004, 0.002
TWIN = STRAT / "research" / "model_calibration" / "width_twin.json"
CONTROL_SHARE = STRAT / "research" / "label_screen" / "thresholds_control_share.json"


def gap(pp, agree) -> float:
    pp, agree = np.asarray(pp, float), np.asarray(agree, bool)
    return float(pp[agree].mean() - pp[~agree].mean())


def simulate(table_of=None) -> pd.DataFrame:
    """The early trades; `table_of(fold)` may return another model's table to read p_buy from."""
    from research.model_calibration.width_sizing import _signal_rows
    from research.model_calibration.width_twin import _gates
    from research.nnnc_framework import entries as E

    share = json.loads(CONTROL_SHARE.read_text())
    base = E.Model("ARO")
    base.params = dataclasses.replace(base.params, stoploss_grace_hours=24.0)
    folds, cuts = sorted(base.folds), json.loads(TWIN.read_text())["models"]["ARO"]["cuts"]
    rows = []
    for i, f in enumerate(folds):
        if f not in EARLY:
            continue
        t = base.tables[f]
        g = _gates(t, E.fold_params(base.prefix, f))
        cut = cuts[folds[i - 1]] if i else cuts[f]
        d = base.simulate(f, g & (t["bb_width"].to_numpy(float) >= cut))
        if d.empty:
            continue
        sr = _signal_rows(t, d)
        bw = sr.merge(pd.DataFrame({"pair": t["pair"].to_numpy(),
                                    "date": pd.DatetimeIndex(t["date"]).tz_convert("UTC")
                                    if pd.DatetimeIndex(t["date"]).tz else
                                    pd.DatetimeIndex(t["date"]).tz_localize("UTC"),
                                    "bb_width": t["bb_width"].to_numpy(float)}),
                      on=["pair", "date"], how="left")["bb_width"].to_numpy()
        r = pd.DataFrame({"fold": f, "pair": d["pair"].to_numpy(), "signal_date": sr["date"],
                          "pp": d["pp"].to_numpy(float), "bb_width": bw,
                          "p_ctl": sr["p_buy"].to_numpy(float)})
        r["agree_ctl_share"] = r["p_ctl"] > float(share[f]["threshold"])
        if table_of is not None:
            tc = table_of(f)
            r["p_model"] = _signal_rows(tc, d)["p_buy"].to_numpy(float)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def check(t: pd.DataFrame) -> dict:
    late = sorted(set(t["fold"]) - set(EARLY))
    if late:
        raise RuntimeError(f"035 reads the early half only; got folds {late}")
    g = gap(t["pp"], t["agree_ctl_share"])
    if len(t) != EXPECTED_TRADES or abs(g - EXPECTED_GAP) > GAP_TOL:
        raise RuntimeError(f"instrument does not reproduce: {len(t)} trades (expect "
                           f"{EXPECTED_TRADES}), control-at-share gap {g:.4f} (expect "
                           f"{EXPECTED_GAP} +- {GAP_TOL})")
    return {"trades": len(t), "control_at_share_gap": round(g, 4)}


def trades(table_of=None) -> pd.DataFrame:
    t = simulate(table_of)
    t.attrs["check"] = check(t)
    return t
