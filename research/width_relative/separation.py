"""033 S1 (T022): does the width-relative model separate trades inside the width book better than
the current model? EARLY half only (analyze I1).

Same instrument as research/candidate_training/co_sizing.py: the width book (gbb rule + bb_width >=
the previous fold's count-matched cut, width_twin.json) is simulated ONCE from the current model's
tables, so both models re-weight the IDENTICAL trades. At each trade's signal bar (entry bar - 1):

  current model  agree = its own entry decision (enter_long of its signal table);
  width-relative agree = buy_argmax AND p_buy > the fold threshold (plan_wr_<m>_sz), i.e. what
                         the sizer arm's tag records.

Reported per model on f00-f07: agree / disagree n and pp per trade, the gap, and the agree_0.5 pp
per unit capital. S1 (pre-registered): new early gap > current early gap (ARO 0.072, GBR 0.063).

The LATE half is refused unless --after-s3 and research/width_relative/verdicts.json records S3 as
judged for that model's exit-off arm: for the exit-off sizer the late gap nearly determines its late
pp, so reading it earlier would read S3 before S2.

    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.width_relative.separation [--after-s3]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

HERE = Path(__file__).resolve().parent
TWIN = STRAT / "research" / "model_calibration" / "width_twin.json"
FOLDS_DIR = STRAT / "evaluation" / "folds"
ARMS = {"WR_ARO": ("wr_aro_sz", "NNNC_WR_ARO", "NNNC_WR_ARO_SZ"),
        "WR_GBR": ("wr_gbr_sz", "NNNC_WR_GBR", "NNNC_WR_GBR_SZ")}
CURRENT_GAP_EARLY = {"ARO": 0.072, "GBR": 0.063}


def wr_agree(t: pd.DataFrame, threshold: float) -> np.ndarray:
    return t["buy_argmax"].to_numpy(bool) & (t["p_buy"].to_numpy(float) > threshold)


def split_stats(pp: np.ndarray, agree: np.ndarray) -> dict:
    pp, agree = np.asarray(pp, float), np.asarray(agree, bool)
    a, d = pp[agree], pp[~agree]
    w = np.where(agree, 1.0, 0.5)
    r = {"trades": int(len(pp)), "unsized_pp": round(float(pp.sum()), 3),
         "agree": {"n": int(len(a)), "pp_per_trade": round(float(a.mean()), 4) if len(a) else None},
         "disagree": {"n": int(len(d)), "pp_per_trade": round(float(d.mean()), 4) if len(d) else None},
         "agree_0.5_pp_per_capital": round(float((pp * w).sum() / w.mean()), 3) if len(pp) else None}
    ag, dg = r["agree"]["pp_per_trade"], r["disagree"]["pp_per_trade"]
    r["gap"] = None if ag is None or dg is None else round(ag - dg, 4)
    return r


def s3_judged(model: str) -> bool:
    p = HERE / "verdicts.json"
    if not p.is_file():
        return False
    v = json.loads(p.read_text()).get("models", {}).get(model, {}).get("exit_off", {})
    return isinstance(v.get("S3"), dict)


def main(argv=None) -> int:
    from research.exit_coupling import feasibility as F
    from research.model_calibration.width_sizing import _signal_rows
    from research.model_calibration.width_twin import _gates
    from research.nnnc_framework import diagnose as D
    from research.nnnc_framework import entries as E

    ap = argparse.ArgumentParser()
    ap.add_argument("--after-s3", action="store_true")
    a = ap.parse_args(argv)
    D.MODELS.update(ARMS)  # runtime registration; the study's own lists are untouched
    tw = json.loads(TWIN.read_text())["models"]
    out = {"label": "033 S1 (pre-registered; early half only unless --after-s3)", "models": {}}
    for m in ("ARO", "GBR"):
        halves = ["early"] + (["late"] if a.after_s3 and s3_judged(m) else [])
        if a.after_s3 and "late" not in halves:
            print(f"{m}: late half REFUSED -- S3 not judged for the exit-off arm", flush=True)
        thr = json.loads((FOLDS_DIR / f"plan_wr_{m.lower()}_sz.json").read_text())["thresholds"]
        base, wr = E.Model(m), E.Model(f"WR_{m}")
        base.params = dataclasses.replace(base.params, stoploss_grace_hours=24.0)
        folds, cuts = sorted(base.folds), tw[m]["cuts"]
        rows = []
        for i, f in enumerate(folds):
            if F.half_of(f) not in halves:
                continue
            t = base.tables[f]
            g = _gates(t, E.fold_params(base.prefix, f))
            cut = cuts[folds[i - 1]] if i else cuts[f]
            d = base.simulate(f, g & (t["bb_width"].to_numpy(float) >= cut))
            if d.empty:
                continue
            cur = _signal_rows(t, d)
            tw_ = wr.tables[f].assign(enter_long=wr_agree(wr.tables[f], float(thr[f])).astype(int))
            new = _signal_rows(tw_, d)
            rows.append(d.assign(half=F.half_of(f), agree_cur=cur["agree"].to_numpy(),
                                 agree_new=new["agree"].to_numpy()))
        res = {}
        if rows:
            allr = pd.concat(rows, ignore_index=True)
            for h, d in allr.groupby("half"):
                res[h] = {"current": split_stats(d["pp"], d["agree_cur"]),
                          "width_relative": split_stats(d["pp"], d["agree_new"])}
        e = res.get("early", {})
        g_new = e.get("width_relative", {}).get("gap")
        res["S1_pass"] = bool(g_new is not None and g_new > CURRENT_GAP_EARLY[m])
        res["current_gap_early_preregistered"] = CURRENT_GAP_EARLY[m]
        out["models"][m] = res
        print(m, json.dumps(res), flush=True)
    name = "separation_after_s3.json" if a.after_s3 else "separation.json"
    (HERE / name).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
