"""ATR-barrier study, stage S2 verdicts (preregistration.json), for the arms that passed S1 and were
backtested: instrument (research/xmodel_sizer/instrument.arm_check, tag check against the arm's own
table) then cs_aro's bar (research/control_sizer/judge.py decide: judge_one PASS vs w_aro AND
wf_aro_g24) and the question, judge_one vs cs_aro. Refuses while the pre-registration is
uncommitted/dirty or a record predates its commit.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.judge
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.control_sizer import judge as CJ  # noqa: E402

HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration.json"


def main() -> int:
    from research.nnnc_switches.judge import judge_one
    from research.xmodel_sizer.instrument import arm_check

    pr = json.loads(PREREG.read_text())
    rel = PREREG.relative_to(STRAT)
    if CJ._git("status", "--porcelain", "--", str(rel)):
        print(f"REFUSED: {rel} has uncommitted changes")
        return 2
    ts = float(CJ._git("log", "-1", "--format=%ct", "--", str(rel)) or 0)
    passed = [a for a, ok in json.loads((HERE / "screen.json").read_text())["s1_pass"].items() if ok]
    bad = [x for a in passed for x in CJ.refusals({**pr, "arm": a}, commit_ts=ts)] if ts else ["not committed"]
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    out = {"label": pr["label"], "arms": {}}
    for arm in passed:
        spec = pr["arms"][arm]
        inst = arm_check(arm, spec, pr)
        vw = vc = vs = None
        if inst["pass"]:
            vw, vc = judge_one(pr, arm, spec["width"]), judge_one(pr, arm, spec["control"])
            vs = judge_one(pr, arm, spec["secondary"])
        r = {"instrument": inst, "decision": CJ.decide(pr, {"pass": inst["pass"]}, vw, vc),
             "question_vs_cs_aro": None if vs is None else {k: vs.get(k) for k in ("verdict", "early_gain_pp", "late_gain_pp")},
             "vs_width": vw, "vs_control": vc, "vs_cs_aro": vs}
        out["arms"][arm] = r
        print(arm, json.dumps({k: r[k] for k in ("decision", "question_vs_cs_aro")}, default=str), flush=True)
    (HERE / "verdicts.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
