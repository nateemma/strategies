"""040 T053: rebuild the NNNC comparators on PURGED training windows (evaluation/foldplan.py purge).

Every fold's train_timerange now ends 1 day before test_start, so every model key changes and every
per-fold artifact derived from the trained model, or keyed by the old train_timerange string, is
re-derived here. The legacy tools hard-code their plans and overwrite fixed outputs
(research/model_calibration/width_twin.json, NNNC_W_*.py, signals/ARO_fNN.parquet), so this driver
reuses their functions and writes its own files. Order (plan.md):

  signals --fold fNN   ARO_PW test-window signal table from wf_aro_g24_pw's fold class (one fold per
                       process; strict model key)          -> signals/ARO_PW_fNN.parquet
  cuts                 width_twin's matched cut per fold (the model's own entry count among the gate-
                       passing bars), then the causal map (fold i uses fold i-1's; f00 its own), keyed
                       by the purged train_timerange       -> width_cuts_aro_pw.json
  build                NNNC_W_ARO_PW / NNNC_CS_ARO_PW / NNNC_CS_ARO_TR_PW (WIDTH_CUTS only; same
                       weights, model key asserted) and the purged plans w_aro_pw (thresholds =
                       wf_aro_g24_pw's), cs_aro_pw (picked next), cs_aro_tr_pw, st_g24_rev_pw
  copy-thresholds      cs_aro_pw's picked thresholds -> cs_aro_tr_pw (cs_aro_tr = cs_aro + tr_scale)

The cut method reproduces the legacy ARO cuts exactly (width_twin.json, NNNC_W_ARO.WIDTH_CUTS) from
the legacy tables; `cuts --legacy-check` re-runs that check.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.train_window.rebuild signals --fold f00
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

HERE = Path(__file__).resolve().parent
SIG = HERE / "signals"
FOLDS = STRAT / "evaluation" / "folds"
CUTS = HERE / "width_cuts_aro_pw.json"
CONTROL = "wf_aro_g24_pw"
CLASSES = {  # new class -> (legacy class it subclasses, module dir)
    "NNNC_W_ARO_PW": "NNNC_W_ARO",
    "NNNC_CS_ARO_PW": "NNNC_CS_ARO",
    "NNNC_CS_ARO_TR_PW": "NNNC_CS_ARO_TR",
}
PLANS = {  # purged plan tag -> (legacy plan, new base class or None to keep the legacy base)
    "w_aro_pw": ("w_aro", "NNNC_W_ARO_PW"),
    "cs_aro_pw": ("cs_aro", "NNNC_CS_ARO_PW"),
    "cs_aro_tr_pw": ("cs_aro_tr", "NNNC_CS_ARO_TR_PW"),
    "st_g24_rev_pw": ("st_g24_rev", None),
}


def _plan(tag: str) -> dict:
    return json.loads((FOLDS / f"plan_{tag}.json").read_text())


def _matched_cut(t: pd.DataFrame, buy: dict) -> float:
    from research.model_calibration.width_twin import _gates

    k = int((t["enter_long"].to_numpy() == 1).sum())
    cand = np.sort(t["bb_width"].to_numpy(float)[_gates(t, buy)])[::-1]
    return float(cand[k - 1]) if 0 < k <= len(cand) else float("inf")


def _buy(cls: str) -> dict:
    return json.loads((STRAT / "NNNC" / f"{cls}.json").read_text())["params"]["buy"]


def signals(fid: str) -> int:
    from research.nnnc_framework.signals import fold_table

    f = next(x for x in _plan(CONTROL)["folds"] if x["fold_id"] == fid)
    SIG.mkdir(exist_ok=True)
    dest = SIG / f"ARO_PW_{fid}.parquet"
    t = fold_table(f["strategy_class"], f["test_timerange"])
    t.to_parquet(dest)
    print(json.dumps({"fold": fid, "class": f["strategy_class"], "rows": len(t),
                      "entries": int((t["enter_long"] == 1).sum()), "out": str(dest.relative_to(STRAT))}))
    return 0


def cuts(legacy_check: bool = False) -> int:
    if legacy_check:
        from research.nnnc_framework import entries as E

        legacy = json.loads((STRAT / "research/model_calibration/width_twin.json").read_text())["models"]["ARO"]["cuts"]
        bad = [f for f in legacy if abs(_matched_cut(pd.read_parquet(E.SIG / f"ARO_{f}.parquet"),
                                                     E.fold_params("NNNC_ARO", f)) - legacy[f]) > 1e-12]
        print("legacy cut mismatches:", bad)
        if bad:
            return 1
    plan = _plan(CONTROL)
    matched = {}
    for f in plan["folds"]:
        fid = f["fold_id"]
        matched[fid] = _matched_cut(pd.read_parquet(SIG / f"ARO_PW_{fid}.parquet"), _buy(f["strategy_class"]))
    causal = {f["train_timerange"]: matched[plan["folds"][i - 1]["fold_id"] if i else "f00"]
              for i, f in enumerate(plan["folds"])}
    CUTS.write_text(json.dumps({"label": "T053 purged windows: width_twin's matched cut on ARO_PW (wf_aro_g24_pw); "
                                         "causal = previous fold's (f00 its own, not causal, as legacy)",
                                "matched": matched, "causal": causal}, indent=1))
    print(json.dumps({"matched": matched}, indent=1))
    return 0


CLASS_TMPL = '''# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""{base} on PURGED training windows (040 T053, research/train_window/rebuild.py; GENERATED): only the
width cuts differ -- keyed by the purged train_timerange, from width_cuts_aro_pw.json (causal). Same
class otherwise; the weights are the purged ARO folds' (model key from MODEL_TRAIN_TIMERANGE)."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from {base} import {base}


class {name}({base}):
    WIDTH_CUTS = {cuts}
'''


def build() -> int:
    from evaluation import foldplan as FP
    from research.nnmt.build_arms import key_of

    causal = json.loads(CUTS.read_text())["causal"]
    lit = json.dumps(causal, indent=8, sort_keys=True)
    for name, base in CLASSES.items():
        src = CLASS_TMPL.format(name=name, base=base, cuts=lit)
        if name == "NNNC_CS_ARO_TR_PW":  # stand-in at each fold's test start = the tables (see the class)
            src += ('    # the stand-in tables = the stand-in fitted at each fold\'s TEST START (bit-identical, 15/15): the right\n'
                    '    # cutoff for a purged fold, whose MODEL_TRAIN_TIMERANGE ends a day earlier and has no fitted stand-in\n'
                    '    TR_SOURCE = "tables"\n')
        (STRAT / "NNNC" / f"{name}.py").write_text(src)
    sys.path.insert(0, str(STRAT / "NNNC"))
    bad = {n: (key_of(n), key_of(b)) for n, b in CLASSES.items() if key_of(n) != key_of(b)}
    if bad:
        raise SystemExit(f"REFUSED: model keys moved: {bad}")
    ctl_thr = _plan(CONTROL).get("thresholds")
    if not ctl_thr or len(ctl_thr) != 15:
        raise SystemExit(f"REFUSED: {CONTROL} has no 15-fold thresholds yet")
    out = {}
    for tag, (src, base) in PLANS.items():
        p = FP.purge(_plan(src), "pw")
        if base:
            p["base_strategy"] = base
        if tag == "w_aro_pw":
            p["thresholds"] = ctl_thr  # as legacy: the width plan inherits the control's thresholds
        p["note"] = f"040 T053: {src} on purged training windows (research/train_window/rebuild.py)"
        errs = FP.validate_plan(p)
        p["validated"], p["validation_errors"] = not errs, errs
        if errs:
            raise SystemExit(f"REFUSED: {tag}: {errs}")
        (FOLDS / f"plan_{tag}.json").write_text(json.dumps(p, indent=1) + "\n")
        out[tag] = {"base": p["base_strategy"], "f00": p["folds"][0]["strategy_class"],
                    "train_timerange": p["folds"][0]["train_timerange"]}
    print(json.dumps(out, indent=1))
    return 0


def copy_thresholds() -> int:
    src = _plan("cs_aro_pw")
    if len(src.get("thresholds", {})) != 15:
        raise SystemExit("REFUSED: cs_aro_pw has no 15-fold thresholds yet")
    p = FOLDS / "plan_cs_aro_tr_pw.json"
    j = json.loads(p.read_text())
    j["thresholds"], j["threshold_picks"] = src["thresholds"], src.get("threshold_picks", {})
    p.write_text(json.dumps(j, indent=1) + "\n")
    print("cs_aro_tr_pw thresholds:", j["thresholds"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("signals", "cuts", "build", "copy-thresholds"))
    ap.add_argument("--fold")
    ap.add_argument("--legacy-check", action="store_true")
    a = ap.parse_args(argv)
    if a.step == "signals":
        return signals(a.fold)
    if a.step == "cuts":
        return cuts(a.legacy_check)
    if a.step == "build":
        return build()
    return copy_thresholds()


if __name__ == "__main__":
    raise SystemExit(main())
