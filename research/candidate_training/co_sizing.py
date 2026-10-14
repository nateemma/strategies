"""032 follow-up 2: the candidate-only model as a SIZER on the width book (2026-10-14; simulator
screen, no training, no backtest; EXPLORATORY).

Same instrument as research/model_calibration/width_sizing.py: the width book (gbb rule + bb_width
>= the previous fold's count-matched cut, from width_twin.json) is simulated ONCE from the base
model's tables, so both sizers re-weight the IDENTICAL trades. At each trade's signal bar (entry
bar - 1) the sizer is read from either the current model's table (<M>) or the candidate-only
model's (CO_<M>):

  agree_0.5   1.0 if that model's own entry fires (enter_long), else 0.5
  rank        0.5 + 0.5 * the trade's p_buy percentile among the fold's width-book trades

READING (declared before the run): per model (ARO, GBR), CO is the better sizer if in BOTH halves
(a) its agree - disagree pp/trade gap exceeds the current model's, AND (b) its agree_0.5 pp per
unit capital exceeds the current model's. Reported for rank as well, but not part of the rule.
EXPLORATORY: the early half of the CO arms was read in 032's S2, and f10-f14 totals were printed.

    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.candidate_training.co_sizing
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

from research.candidate_training.instrument import ARMS  # noqa: E402
from research.exit_coupling import feasibility as F  # noqa: E402
from research.model_calibration.width_sizing import _signal_rows  # noqa: E402
from research.model_calibration.width_twin import _gates  # noqa: E402
from research.nnnc_framework import diagnose as D  # noqa: E402
from research.nnnc_framework import entries as E  # noqa: E402


HERE = Path(__file__).resolve().parent
TWIN = STRAT / "research" / "model_calibration" / "width_twin.json"


def _sized(d: pd.DataFrame) -> dict:
    """agree/disagree split and sized books for one half; d has pp, agree, p_rank."""
    ag, dis = d[d["agree"]], d[~d["agree"]]
    base = float(d["pp"].sum())
    r = {"trades": len(d), "unsized_pp": round(base, 3),
         "agree": {"n": len(ag), "pp_per_trade": round(float(ag["pp"].mean()), 4) if len(ag) else None},
         "disagree": {"n": len(dis), "pp_per_trade": round(float(dis["pp"].mean()), 4) if len(dis) else None}}
    for sch, w in (("agree_0.5", np.where(d["agree"], 1.0, 0.5)), ("rank", 0.5 + 0.5 * d["p_rank"])):
        pp, cap = float((d["pp"] * w).sum()), float(np.mean(w))
        r[sch] = {"pp": round(pp, 3), "capital": round(cap, 4), "pp_per_capital": round(pp / cap, 3)}
    a, b = r["agree"]["pp_per_trade"], r["disagree"]["pp_per_trade"]
    r["gap"] = None if a is None or b is None else round(a - b, 4)
    return r


def main() -> int:
    D.MODELS.update(ARMS)  # runtime registration, as in instrument.py
    tw = json.loads(TWIN.read_text())["models"]
    out = {"label": "EXPLORATORY: CO vs current model as sizer on the identical width book",
           "models": {}}
    for m in ("ARO", "GBR"):
        base, co = E.Model(m), E.Model(f"CO_{m}")
        base.params = dataclasses.replace(base.params, stoploss_grace_hours=24.0)
        folds, cuts = sorted(base.folds), tw[m]["cuts"]
        rows = []
        for i, f in enumerate(folds):
            t = base.tables[f]
            g = _gates(t, E.fold_params(base.prefix, f))
            cut = cuts[folds[i - 1]] if i else cuts[f]
            d = base.simulate(f, g & (t["bb_width"].to_numpy(float) >= cut))
            if d.empty:
                continue
            d = d.assign(half=F.half_of(f))
            for who, mdl in (("cur", base), ("co", co)):
                s = _signal_rows(mdl.tables[f], d)
                d[f"agree_{who}"] = s["agree"].to_numpy()
                d[f"p_{who}"] = s["p_buy"].to_numpy()
                d[f"rank_{who}"] = d[f"p_{who}"].rank(pct=True)
            rows.append(d)
        a = pd.concat(rows, ignore_index=True)
        res = {}
        for h, d in a.groupby("half"):
            res[h] = {who: _sized(d.assign(agree=d[f"agree_{who}"], p_rank=d[f"rank_{who}"]))
                      for who in ("cur", "co")}
        better = all(res[h]["co"]["gap"] is not None and res[h]["cur"]["gap"] is not None
                     and res[h]["co"]["gap"] > res[h]["cur"]["gap"]
                     and res[h]["co"]["agree_0.5"]["pp_per_capital"]
                     > res[h]["cur"]["agree_0.5"]["pp_per_capital"] for h in ("early", "late"))
        res["co_better_sizer"] = better
        out["models"][m] = res
        print(m, json.dumps(res), flush=True)
    (HERE / "co_sizing.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
