"""038 vote C as triggers: the pre-registered verdicts (research/holt_trigger/preregistration.json).

Per arm: instrument (trigger causality PASS, trades >= min_arm_trades, no zero-trade half), then
  - alone: judge_one(arm, the arm's family comparator: st_g24_mom for up, st_fam_gbb for the fade);
  - beside the width book: [w_aro, arm] through one shared wallet vs w_aro alone.
The same code as research/pair_triggers/judge.py, with a per-arm comparator. Refuses while the
pre-registration is uncommitted/dirty or any arm record predates its commit.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.holt_trigger.judge
"""
from __future__ import annotations

import json
from pathlib import Path

from research.pair_triggers import judge as PJ

HERE = Path(__file__).resolve().parent
PJ.PREREG, PJ.HERE = HERE / "preregistration.json", HERE


def main() -> int:
    from research.nnnc_switches.judge import judge_one

    pr = json.loads(PJ.PREREG.read_text())
    bad = PJ.refusals(pr)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    out = {"label": pr["label"], "arms": {}}
    for arm, spec in pr["arms"].items():
        inst = PJ.instrument(pr, arm, spec["trigger"])
        r = {"instrument": inst}
        if inst["pass"]:
            r["alone"] = judge_one(pr, arm, spec["comparator"])
            r["beside_width_book"] = PJ.beside_width(arm)
            r["decision"] = ("candidate trigger" if "PASS" in (r["alone"]["verdict"], r["beside_width_book"]["verdict"])
                             else "not a candidate")
        else:
            r["decision"] = "INCONCLUSIVE (instrument)"
        out["arms"][arm] = r
        a, b = r.get("alone", {}), r.get("beside_width_book", {})
        print(arm, r["decision"], f"| alone vs {spec['comparator']}", a.get("verdict"), a.get("arm_pp"), "vs",
              a.get("ctl_pp"), a.get("early_gain_pp"), a.get("late_gain_pp"), (a.get("paired") or {}).get("verdict"),
              "| beside w_aro", b.get("verdict"), b.get("early_gain_pp"), b.get("late_gain_pp"),
              (b.get("paired_whole") or {}).get("verdict"), "| trades", inst["trades"], flush=True)
    (HERE / "verdicts.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
