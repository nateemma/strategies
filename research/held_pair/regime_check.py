"""031 follow-up: does a held-pair add earn differently by market regime? (cheap gate before any
regime-gated add; constitution VII.)

Instrument (fixed, not searched): 026's `volreg` -- cross-pair median 1-day volatility vs its
30-day median, "off" (calm) below `research.nnnc_framework.entries.VOLREG_CUT` (1.366x, the 026
tighter cut used by the NNNC framework study). It is the only regime instrument that passed a
feasibility gate in this repo (SimpleTrigger reversion), and the one already wired into the
framework (`REVERSION_REGIME_GATE`).

Per add in the screen's DCA simulation (24 h grace, last fill >= 1 h): the add's marginal pp =
that trade's pp in the DCA book minus the same trade's pp (same pair, same entry bar) in the
no-add book. Adds whose trade has no counterpart in the no-add book (re-sequenced) are counted
and excluded. Each add is tagged by the volreg state on its signal bar.

DECISION RULE (declared here before the run): a regime gate on the add is worth building only if,
for a variant, the "off" state's marginal add pp has the SAME SIGN in both halves, OPPOSITE to the
"on" state's, on at least 3 of the 4 models. Otherwise there is nothing to gate on.

    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.held_pair.regime_check
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

from research.exit_coupling import feasibility as F  # noqa: E402
from research.held_pair.screen import simulate  # noqa: E402
from research.nnnc_framework import entries as E  # noqa: E402


HERE = Path(__file__).resolve().parent
GRACE = 24.0
VARIANTS = (("all", 1.0), ("profit", 1.0))


def main() -> int:
    vr = E.volreg_on_dates()
    out = {"instrument": "volreg (026), off when < %.3f" % E.VOLREG_CUT, "grace": GRACE,
           "rule": "off-state marginal add pp same sign both halves and opposite to on-state, "
                   "on >= 3/4 models", "models": {}}
    for m in ("ARO", "GBR", "GFM", "MLX"):
        model = E.Model(m)
        params = dataclasses.replace(model.params, stoploss_grace_hours=GRACE)
        res = {}
        for cond, frac in VARIANTS:
            cell = {h: {s: {"n": 0, "pp": 0.0} for s in ("on", "off", "nan")}
                    for h in ("early", "late")}
            unpaired = 0
            for f in model.folds:
                mask = model.tables[f]["enter_long"].to_numpy() == 1
                base = simulate(model, f, mask, params, "NONE", None, 0.0)
                dca = simulate(model, f, mask, params, "DCA", cond, 1.0, add_frac=frac)
                if dca.empty:
                    continue
                bpp = {(t.pair, t.e): t.pp for t in base.itertuples()} if not base.empty else {}
                h = F.half_of(f)
                for t in dca.itertuples():
                    if t.add_e < 0:
                        continue
                    k = (t.pair, t.e)
                    if k not in bpp:
                        unpaired += 1
                        continue
                    _, dates = F.bars_for(t.pair)
                    sig = dates[t.add_e - 1]
                    v = vr.reindex(pd.DatetimeIndex([sig]), method="ffill").iloc[0]
                    s = "nan" if np.isnan(v) else ("off" if v < E.VOLREG_CUT else "on")
                    cell[h][s]["n"] += 1
                    cell[h][s]["pp"] += float(t.pp - bpp[k])
            for h in cell:
                for s in cell[h]:
                    c = cell[h][s]
                    c["pp"] = round(c["pp"], 3)
                    c["pp_per_add"] = round(c["pp"] / c["n"], 4) if c["n"] else None
            off_e, off_l = cell["early"]["off"]["pp"], cell["late"]["off"]["pp"]
            on_e, on_l = cell["early"]["on"]["pp"], cell["late"]["on"]["pp"]
            separates = bool(np.sign(off_e) == np.sign(off_l) != 0
                             and np.sign(on_e) == np.sign(on_l) == -np.sign(off_e))
            res[f"{cond}:{frac:g}"] = {"cells": cell, "unpaired_adds": unpaired,
                                       "separates": separates}
            print(m, cond, json.dumps({h: {s: cell[h][s] for s in ("on", "off")} for h in cell}),
                  "unpaired", unpaired, "separates", separates, flush=True)
        out["models"][m] = res
    verdict = {}
    for v in (f"{c}:{f:g}" for c, f in VARIANTS):
        n = sum(out["models"][m][v]["separates"] for m in out["models"])
        verdict[v] = {"models_separating": n, "worth_gating": n >= 3}
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict))
    (HERE / "regime_check.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
