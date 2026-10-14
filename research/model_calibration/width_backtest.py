"""Confirm the width-book finding with real backtests (2026-10-13; research/model_calibration).

Per model: the width book `w_<m>` (NNNC_W_<M>: the gbb rule + bb_width >= the previous fold's cut;
the model loads but does not select; held-pair add off) against the model's own 24 h control book
(add off), over f00-f14:
  * early / late gain (sum of per-fold return_pct), trades, paired and risk verdicts;
  * simulator fidelity on this NON-model entry mask: (pair, open date) recall / precision of the
    rebuilt width book against the backtest, and simulated vs backtested pp per half.

Labels: DIAGNOSTIC. Both halves were seen in the simulator before this run, and f00's cut is not
causal (it uses f00's own count).

    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.model_calibration.width_backtest
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
from research.model_calibration.width_twin import _gates  # noqa: E402
from research.nnnc_framework import entries as E  # noqa: E402


HERE = Path(__file__).resolve().parent
FOLDS = STRAT / "evaluation" / "folds"
CONTROLS = {"ARO": "wf_aro_g24", "GBR": "fs_gbr_gr", "GFM": "fs_gfm_gr", "MLX": "fs_mlx_gr"}


def _plan(tag: str) -> dict:
    return json.loads((FOLDS / f"plan_{tag}.json").read_text())


def main() -> int:
    from evaluation import risk
    from evaluation.compare_pooled import compare_data
    from evaluation.pool import pool

    tw = json.loads((HERE / "width_twin.json").read_text())["models"]
    out = {"label": "DIAGNOSTIC -- PROVISIONAL (simulator results seen first; f00 cut not causal)",
           "models": {}}
    for m, ctl in CONTROLS.items():
        ap, cp = pool(_plan(f"w_{m.lower()}")), pool(_plan(ctl))
        if ap["folds_missing"] or cp["folds_missing"]:
            out["models"][m] = {"missing": ap["folds_missing"] + cp["folds_missing"]}
            continue
        early = {f"f{i:02d}" for i in range(8)}
        s = {h: {k: sum(f["return_pct"] or 0.0 for f in p["per_fold"]
                        if (f["fold_id"] in early) == (h == "early"))
                 for k, p in (("arm", ap), ("ctl", cp))} for h in ("early", "late")}
        cmp = compare_data(ap, cp)
        rk = risk.compare(ap["trades"], cp["trades"])
        # simulator fidelity on the width mask
        model = E.Model(m)
        model.params = dataclasses.replace(model.params, stoploss_grace_hours=24.0)
        folds = sorted(model.folds)
        cuts = tw[m]["cuts"]
        sim_keys, sim_pp = set(), {"early": 0.0, "late": 0.0}
        for i, f in enumerate(folds):
            t = model.tables[f]
            g = _gates(t, E.fold_params(model.prefix, f))
            cut = cuts[folds[i - 1]] if i else cuts[f]
            d = model.simulate(f, g & (t["bb_width"].to_numpy(float) >= cut))
            if d.empty:
                continue
            sim_pp[F.half_of(f)] += float(d["pp"].sum())
            for r in d.itertuples():
                _, dates = F.bars_for(r.pair)
                sim_keys.add((f, r.pair, pd.Timestamp(dates[r.e]).tz_convert("UTC")))
        bt_keys = {(t["fold_id"], t["pair"], pd.Timestamp(t["open_date"]).tz_convert("UTC")
                    if pd.Timestamp(t["open_date"]).tz else pd.Timestamp(t["open_date"], tz="UTC"))
                   for t in ap["trades"]}
        both = sim_keys & bt_keys
        out["models"][m] = {
            "width": {h: round(s[h]["arm"], 3) for h in s},
            "control": {h: round(s[h]["ctl"], 3) for h in s},
            "gain": {h: round(s[h]["arm"] - s[h]["ctl"], 3) for h in s},
            "trades": {"width": ap["n_trades_pooled"], "control": cp["n_trades_pooled"]},
            "paired": {"verdict": cmp["verdict"], "p": (cmp.get("bootstrap") or {}).get("p")},
            "risk": {"sharpe": rk["verdict_sharpe"], "maxdd": rk["verdict_maxdd"],
                     "sharpe_arm_ctl": [round(rk["arm"]["sharpe"], 3), round(rk["ctl"]["sharpe"], 3)],
                     "maxdd_arm_ctl": [round(rk["arm"]["maxdd"], 3), round(rk["ctl"]["maxdd"], 3)]},
            "simulator": {"recall": round(len(both) / max(len(bt_keys), 1), 4),
                          "precision": round(len(both) / max(len(sim_keys), 1), 4),
                          "sim_pp": {h: round(v, 3) for h, v in sim_pp.items()},
                          "simulated_trades": len(sim_keys)},
        }
        print(m, json.dumps(out["models"][m]), flush=True)
    (HERE / "width_backtest.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
