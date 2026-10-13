"""NNNC framework study: choose the stage-2 candidates from the stage-1 EARLY-half screens.

Fixed (committed) before any lever result was read. Inputs: `diagnose.json` (exit/stop levers,
027 replay) and `entries.json` (entry levers, signal tables + simulator); both hold EARLY-half
(f00-f07) deltas only, and only for models whose replay passed its fidelity gate.

Rule, per lever value (base values excluded):
  * QUALIFIES if its early delta is >= +MIN_PRIMARY_PP on ARO (arm D, primary) AND > 0 on at
    least REPS_NEEDED of the replication models screened for that lever (GBR, GFM, MLX;
    `guard_on` exists only on the three guards-open models, so it needs 1 of GBR/GFM;
    `guard_off` exists only on MLX and cannot qualify -- no primary);
  * per lever, the qualifying value with the largest early delta summed over the screened models
    is the lever's candidate;
  * at most MAX_CANDIDATES levers go forward, ranked by that summed delta.
A lever whose replay could not pass fidelity on ARO is not eligible.

Stage 2 (walk-forward arms on every model, judged under preregistration.json) tests these values
on the LATE half (f08-f14), which no screen has touched.

    PYTHONPATH=. ../../.venv/bin/python -m research.nnnc_framework.select
"""

from __future__ import annotations

import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
PRIMARY, REPS = "ARO", ("GBR", "GFM", "MLX")
MIN_PRIMARY_PP = 0.5
REPS_NEEDED = 2
MAX_CANDIDATES = 6
#: NNNC_FS_ROUND: round 2 reads diagnose_r2/entries_r2 and writes selection_r2, same rule;
#: its grace base is 24 h (adopted in round 1).
ROUND = os.environ.get("NNNC_FS_ROUND", "1")
SUFFIX = "" if ROUND == "1" else f"_r{ROUND}"
BASE_VALUES = {"take_profit": "0.022", "tp_atr_k": "base", "roi": "base",
               "grace_hours": "6.5" if ROUND == "1" else "24.0",
               "grace_level": "-0.27", "stop_scale": "1.0", "vol_confirm": "True",
               "unclog_12h_hours": "base", "unclog_1d_hours": "24.0", "max_hold_days": "10",
               "metric_overbought": "0.98"}


def screens() -> dict:
    """{lever: {value: {model: early_pp}}} over both screens."""
    out: dict = {}
    diag = json.loads((HERE / f"diagnose{SUFFIX}.json").read_text())["models"]
    for m, d in diag.items():
        for lever, vals in (d.get("levers_early") or {}).items():
            for v, r in vals.items():
                if v != BASE_VALUES.get(lever):
                    out.setdefault(f"exit:{lever}", {}).setdefault(v, {})[m] = r["early_pp"]
    ent = json.loads((HERE / f"entries{SUFFIX}.json").read_text())
    for m, d in ent.items():
        for lever, vals in (d.get("levers_early") or {}).items():
            if lever == "base":
                continue
            for v, r in vals.items():
                out.setdefault(f"entry:{lever}", {}).setdefault(v, {})[m] = r["early_pp"]
    return out


def choose(sc: dict) -> dict:
    cands, table = [], {}
    for lever, vals in sc.items():
        best = None
        for v, by in vals.items():
            reps = [r for r in REPS if r in by]
            need = 1 if lever == "entry:guard_on" else REPS_NEEDED
            ok = (PRIMARY in by and by[PRIMARY] >= MIN_PRIMARY_PP
                  and sum(by[r] > 0 for r in reps) >= need and len(reps) >= need)
            total = sum(by.values())
            table.setdefault(lever, {})[v] = {"by_model": by, "sum": round(total, 3),
                                              "qualifies": ok}
            if ok and (best is None or total > best[1]):
                best = (v, total)
        if best:
            cands.append({"lever": lever, "value": best[0], "early_sum_pp": round(best[1], 3)})
    cands.sort(key=lambda c: -c["early_sum_pp"])
    return {"candidates": cands[:MAX_CANDIDATES], "not_taken": cands[MAX_CANDIDATES:],
            "table": table}


def main() -> int:
    res = choose(screens())
    (HERE / f"selection{SUFFIX}.json").write_text(json.dumps(res, indent=1))
    for c in res["candidates"]:
        print(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
