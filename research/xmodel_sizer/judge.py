"""Cross-model sizer study: the pre-registered verdicts (preregistration.json), per arm.

Refuses while the pre-registration is uncommitted/dirty or any arm record predates its commit. Per arm:
instrument must pass (else INCONCLUSIVE, no P&L read); primary = cs_aro's bar (research/control_sizer/
judge.py `decide`: judge_one PASS vs the model's width book AND its control, small-margin rule);
secondary judge_one vs the arm's `secondary` comparator (mix arms: the plain sizer).

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.xmodel_sizer.judge
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


def refusals(pr: dict) -> list[str]:
    rel = PREREG.relative_to(STRAT)
    out = []
    if CJ._git("status", "--porcelain", "--", str(rel)):
        out.append(f"{rel} has uncommitted changes")
    ts = CJ._git("log", "-1", "--format=%ct", "--", str(rel))
    if not ts:
        return out + [f"{rel} is not committed"]
    for arm in pr["arms"]:
        out += CJ.refusals({**pr, "arm": arm}, commit_ts=float(ts))
    return out


def main() -> int:
    from research.nnnc_switches.judge import judge_one

    pr = json.loads(PREREG.read_text())
    bad = refusals(pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    inst = json.loads((HERE / "instrument.json").read_text())
    out = {"label": pr["label"], "arms": {}}
    for arm, spec in pr["arms"].items():
        ia = {"pass": inst[arm]["pass"]}
        vw = vc = vs = None
        if ia["pass"]:
            vw, vc = judge_one(pr, arm, spec["width"]), judge_one(pr, arm, spec["control"])
            vs = judge_one(pr, arm, spec["secondary"])
        r = {"decision": CJ.decide(pr, ia, vw, vc),
             "secondary": None if vs is None else {"vs": spec["secondary"], "verdict": vs["verdict"],
                                                   "early_gain_pp": vs.get("early_gain_pp"),
                                                   "late_gain_pp": vs.get("late_gain_pp")},
             "vs_width": vw, "vs_control": vc, "vs_secondary": vs}
        out["arms"][arm] = r
        print(arm, json.dumps({k: r[k] for k in ("decision", "secondary")}, default=str), flush=True)
    (HERE / "verdicts.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
