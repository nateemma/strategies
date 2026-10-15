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
#: GBR extension: the control sizer's 0.5-share thresholds (plan_cs_gbr.json) and the reproduction
#: values recorded in preregistration_gbr.json before any GBR input score.
CS_GBR_PLAN = STRAT / "evaluation" / "folds" / "plan_cs_gbr.json"
PREREG_GBR = Path(__file__).resolve().parent / "preregistration_gbr.json"


def _share_thresholds(model: str) -> dict:
    if model == "ARO":
        return {f: v["threshold"] for f, v in json.loads(CONTROL_SHARE.read_text()).items()}
    return dict(json.loads(CS_GBR_PLAN.read_text()).get("thresholds", {}))


def expected(model: str) -> tuple[int, float] | None:
    if model == "ARO":
        return EXPECTED_TRADES, EXPECTED_GAP
    if PREREG_GBR.is_file():
        r = json.loads(PREREG_GBR.read_text())["reproduce"]
        return r["trades"], r["control_at_share_gap"]
    return None


def gap(pp, agree) -> float:
    pp, agree = np.asarray(pp, float), np.asarray(agree, bool)
    return float(pp[agree].mean() - pp[~agree].mean())


def simulate(table_of=None, model: str = "ARO") -> pd.DataFrame:
    """The early trades; `table_of(fold)` may return another model's table to read p_buy from."""
    from research.model_calibration.width_sizing import _signal_rows
    from research.model_calibration.width_twin import _gates
    from research.nnnc_framework import entries as E

    share = _share_thresholds(model)
    base = E.Model(model)
    base.params = dataclasses.replace(base.params, stoploss_grace_hours=24.0)
    folds, cuts = sorted(base.folds), json.loads(TWIN.read_text())["models"][model]["cuts"]
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
        r["agree_ctl_share"] = r["p_ctl"] > float(share[f]) if f in share else np.nan
        if table_of is not None:
            tc = table_of(f)
            r["p_model"] = _signal_rows(tc, d)["p_buy"].to_numpy(float)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def check(t: pd.DataFrame, exp: tuple[int, float] | None = (EXPECTED_TRADES, EXPECTED_GAP)) -> dict:
    """Early folds only; and, when `exp` is given, the trade count and control-at-share gap must
    reproduce. `exp=None` only measures (the GBR values recorded before its pre-registration)."""
    late = sorted(set(t["fold"]) - set(EARLY))
    if late:
        raise RuntimeError(f"035 reads the early half only; got folds {late}")
    g = gap(t["pp"], t["agree_ctl_share"].astype(bool))
    if exp is not None and (len(t) != exp[0] or abs(g - exp[1]) > GAP_TOL):
        raise RuntimeError(f"instrument does not reproduce: {len(t)} trades (expect "
                           f"{exp[0]}), control-at-share gap {g:.4f} (expect "
                           f"{exp[1]} +- {GAP_TOL})")
    return {"trades": len(t), "control_at_share_gap": round(g, 4)}


def trades(table_of=None, model: str = "ARO", measure_only: bool = False) -> pd.DataFrame:
    t = simulate(table_of, model)
    exp = None if measure_only else expected(model)
    if exp is None and not measure_only:
        raise RuntimeError(f"{model}: no recorded reproduction values (preregistration_gbr.json)")
    t.attrs["check"] = check(t, exp)
    return t
