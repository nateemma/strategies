"""Cross-model sizer study: instrument checks per arm (cs_aro's checks via research/hgb_sizer/instrument
fold_checks, tag check against the arm's own score table; entries without a score <= 1%). No P&L read.

    PYTHONPATH=. ../../.venv/bin/python -m research.xmodel_sizer.instrument
"""
from __future__ import annotations

import json
from pathlib import Path

from research.hgb_sizer import instrument as HI
from research.xmodel_sizer import knots as K

HERE = Path(__file__).resolve().parent


def arm_check(arm: str, spec: dict, pr: dict) -> dict:
    from evaluation.fingerprint import load_strategy_class
    from research.control_sizer.instrument import _plan
    from research.label_screen.instrument import sz_arm_summary

    HI.OUT = K.out_dir(spec["model"]) if arm.endswith("_mix") else K.link_plain(spec["model"])
    plan, wplan = _plan(arm), _plan(spec["width"])
    cuts = load_strategy_class(plan["base_strategy"]).WIDTH_CUTS
    thr = plan.get("thresholds", {})
    folds = {f["fold_id"]: HI.fold_checks(f, fw, thr, cuts) for f, fw in zip(plan["folds"], wplan["folds"])}
    out = {"folds": folds, "halves": {}}
    for h in ("early", "late"):
        s = sz_arm_summary({k: folds[k] for k in pr[h]})
        n = sum(folds[k]["n_trades"] for k in pr[h])
        miss = sum(folds[k]["entries_without_hgb_p"] for k in pr[h])
        s["missing_score_share"] = round(miss / n, 4) if n else None
        s["missing_score_ok"] = bool(n) and miss / n <= HI.MAX_MISSING
        out["halves"][h] = s
    out["pass"] = all(s["S0_pass"] and not s["S0_inconclusive"] and s["missing_score_ok"]
                      for s in out["halves"].values())
    return out


def main() -> int:
    pr = json.loads((HERE / "preregistration.json").read_text())
    res = {}
    for arm, spec in pr["arms"].items():
        res[arm] = arm_check(arm, spec, pr)
        print(arm, "instrument pass:", res[arm]["pass"], json.dumps(res[arm]["halves"]), flush=True)
    (HERE / "instrument.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
