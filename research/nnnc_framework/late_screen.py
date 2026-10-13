"""Reported only (computed after the pre-registration commit): the chosen values' screen deltas on
BOTH halves, to set beside the walk-forward arms (does the screen predict the arm?).

    PYTHONPATH=. ../../.venv/bin/python -m research.nnnc_framework.late_screen
    NNNC_FS_ROUND=2 selects round 2 (selection_r2.json -> late_screen_r2.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

from research.exit_coupling import feasibility as F  # noqa: E402
from research.exit_coupling import paths as P  # noqa: E402
from research.nnnc_framework import diagnose as D  # noqa: E402
from research.nnnc_framework import entries as E  # noqa: E402


HERE = Path(__file__).resolve().parent


def exit_deltas(m: str, lever: str, value: str) -> dict:
    tag, cls_name, prefix = D.MODELS[m]
    D._JSON_FOR[cls_name] = STRAT / "NNNC" / f"{prefix}_f00.json"
    df, _ = F.trade_rows(tag, cls_name, D.FAMILY)
    cls, params, roi = D.exit_setup(cls_name)
    params = F.family_params(cls, params, D.FAMILY)
    vals = D.LEVERS[lever.split(":")[1]]
    key = next(k for k in vals if str(k) == value)
    pover, rover = vals[key]
    p = D.lever_params(params, pover)
    out = {"early": 0.0, "late": 0.0}
    for t in df.itertuples():
        b, _ = F.bars_for(t.pair)
        cf = P.replay(b, t.e, t.open_rate, p, roi=roi, roi_suppressed=t.roi_suppressed, end=t.end,
                      rules=P.Rules(**rover), tick=t.tick)
        out[t.half] += (cf.profit - t.null_profit) * t.stake * t.scale
    return {h: round(v, 3) for h, v in out.items()}


def thr_deltas(m: str, d: float) -> dict:
    model = E.Model(m)
    out = {"early": 0.0, "late": 0.0}
    for f, t in model.tables.items():
        buy = E.fold_params(model.prefix, f)
        base = model.simulate(f, t["enter_long"].to_numpy() == 1)["pp"].sum()
        arm = model.simulate(f, E.base_mask(
            t, buy, thr=round(float(buy["prediction_threshold"]), 2) + d))["pp"].sum()
        out["early" if f in F.EARLY else "late"] += float(arm - base)
    return {h: round(v, 3) for h, v in out.items()}


def main() -> int:
    sel = json.loads((HERE / f"selection{D.SUFFIX}.json").read_text())["candidates"]
    res = {}
    for c in sel:
        res[c["lever"]] = {}
        for m in D.MODELS:
            if c["lever"] == "entry:thr_shift":
                res[c["lever"]][m] = thr_deltas(m, float(c["value"]))
            elif m != "MLX":  # MLX failed the exit-replay fidelity gate
                res[c["lever"]][m] = exit_deltas(m, c["lever"], c["value"])
        print(c["lever"], res[c["lever"]], flush=True)
    (HERE / f"late_screen{D.SUFFIX}.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
