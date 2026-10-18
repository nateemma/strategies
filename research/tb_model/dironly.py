"""ATR-barrier study, direction-only arm (preregistration_dironly.json): S1 and S2 verdicts for cs_tbd
(arm A trained on barrier-hit windows only; score p_buy). ONE step per process:

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.dironly screen
    ... dironly judge
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.tb_model.dir_arms import TARGET, _tables  # noqa: E402

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_dironly.json"
ARM = "cs_tbd"


def screen() -> dict:
    from research.indicator_screen import scores as S
    from research.indicator_screen import windows as W
    from research.indicator_screen.panel import Panel

    panel = Panel(tradeable_only=True)
    y = panel.columns([TARGET])[TARGET].astype(float)
    sc = {"ARO": _tables(panel, "ARO", False), ARM: _tables(panel, "CS_TBD", False),
          f"{ARM}_dir": _tables(panel, "CS_TBD", True), "cs_tba_d": _tables(panel, "CS_TBA", True)}
    rows = {}
    for f in W.folds():
        base = f.mask(panel.date) & panel.data_ok & np.isfinite(y) & np.all([np.isfinite(v) for v in sc.values()], axis=0)
        r = {"half": f.half}
        for pop, m in (("cand", base & panel.candidate), ("all", base)):
            i = np.flatnonzero(m)
            for k, p in sc.items():
                r[f"{k}_{pop}_auc"] = S.auc(p[i], y[i])
        rows[f.id] = r
    summary = {h: {k: float(np.nanmedian([r[k] for r in rows.values() if r["half"] == h]))
                   for k in rows["f00"] if k != "half"} for h in ("early", "late")}
    s1 = {ARM: all(summary[h][f"{ARM}_cand_auc"] > summary[h]["ARO_cand_auc"] for h in ("early", "late"))}
    out = {"label": "S1 (preregistration_dironly.json); cs_tbd_dir and cs_tba_d reported only",
           "summary": summary, "s1_pass": s1, "folds": rows}
    (HERE / "screen_dironly.json").write_text(json.dumps(out, indent=1))
    for h in ("early", "late"):
        print(h, {k: round(v, 3) for k, v in summary[h].items()})
    print("S1 pass:", s1)
    return out


def judge() -> int:
    from research.control_sizer import judge as CJ
    from research.nnnc_switches.judge import judge_one
    from research.xmodel_sizer.instrument import arm_check

    pr = json.loads(PREREG.read_text())
    rel = PREREG.relative_to(STRAT)
    if CJ._git("status", "--porcelain", "--", str(rel)):
        print(f"REFUSED: {rel} has uncommitted changes")
        return 2
    ts = float(CJ._git("log", "-1", "--format=%ct", "--", str(rel)) or 0)
    if not json.loads((HERE / "screen_dironly.json").read_text())["s1_pass"][ARM]:
        print("S1 failed: no S2")
        return 0
    bad = CJ.refusals({**pr, "arm": ARM}, commit_ts=ts) if ts else ["not committed"]
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    spec = pr["arms"][ARM]
    inst = arm_check(ARM, spec, pr)
    vw = vc = vs = v2 = None
    if inst["pass"]:
        vw, vc = judge_one(pr, ARM, spec["width"]), judge_one(pr, ARM, spec["control"])
        vs = judge_one(pr, ARM, spec["secondary"])
        if (STRAT / "evaluation" / "folds" / "results_cs_tba_d_seed1.json").exists():
            v2 = judge_one(pr, ARM, "cs_tba_d")
    pick = lambda v: None if v is None else {k: v.get(k) for k in ("verdict", "early_gain_pp", "late_gain_pp")}
    r = {"label": pr["label"], "instrument": inst, "decision": CJ.decide(pr, {"pass": inst["pass"]}, vw, vc),
         "question_vs_cs_aro": pick(vs), "vs_cs_tba_d": pick(v2),
         "vs_width": vw, "vs_control": vc, "vs_cs_aro": vs}
    print(ARM, "instrument", inst["pass"], json.dumps({k: r[k] for k in ("decision", "question_vs_cs_aro", "vs_cs_tba_d")}, default=str))
    (HERE / "verdicts_dironly.json").write_text(json.dumps(r, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("screen", "judge"))
    a = ap.parse_args(argv)
    if a.step == "screen":
        screen()
        return 0
    return judge()


if __name__ == "__main__":
    raise SystemExit(main())
