"""HGB sizer study: the pre-registered verdict for `cs_hgb` (preregistration.json).

  1. Refuses while the pre-registration is uncommitted or dirty, or if any `cs_hgb` record is older
     than its commit.
  2. Instrument (instrument.json) must pass in both halves; otherwise INCONCLUSIVE, no P&L read.
  3. Primary (cs_aro's bar, research/control_sizer/judge.py `decide`): judge_one against `w_aro` AND
     `wf_aro_g24`; a half gain under small_margin_pp makes a PASS INCONCLUSIVE.
  4. Secondary: judge_one against the ARO sizer `cs_aro`, reported beside the decision.

    PYTHONPATH=. ../../.venv/bin/python -m research.hgb_sizer.judge
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
    return out + CJ.refusals(pr, commit_ts=float(ts))


def main() -> int:
    from research.nnnc_switches.judge import judge_one

    pr = json.loads(PREREG.read_text())
    bad = refusals(pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    inst = json.loads((HERE / "instrument.json").read_text())
    vw = vc = vs = None
    if inst.get("pass"):
        vw = judge_one(pr, pr["arm"], pr["comparators"]["width"])
        vc = judge_one(pr, pr["arm"], pr["comparators"]["control"])
        vs = judge_one(pr, pr["arm"], pr["comparators"]["incumbent_sizer"])
    out = {"label": pr["label"], "decision": CJ.decide(pr, inst, vw, vc),
           "secondary_vs_cs_aro": None if vs is None else {
               "verdict": vs["verdict"], "early_gain_pp": vs.get("early_gain_pp"),
               "late_gain_pp": vs.get("late_gain_pp")},
           "vs_width": vw, "vs_control": vc, "vs_cs_aro": vs}
    print(json.dumps({k: out[k] for k in ("decision", "secondary_vs_cs_aro")}, indent=1, default=str))
    (HERE / "verdicts.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
