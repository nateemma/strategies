"""mofn_4of8 as a trigger: the pre-registered verdicts. Same rules and code as
research/pair_triggers/judge.py, pointed at this folder's pre-registration: preregistration.json
(plain exits, verdicts.json) or, with --c2, preregistration_c2.json (C2 exit, c2/verdicts.json, plus
the secondary judge_one(st_mofn48_c2, st_mofn48) in c2/secondary.json).

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.mofn_trigger.judge [--c2]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from research.pair_triggers import judge as PJ

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    if "--c2" in sys.argv:
        PJ.PREREG, PJ.HERE = HERE / "preregistration_c2.json", HERE / "c2"
        PJ.HERE.mkdir(exist_ok=True)
        rc = PJ.main()
        if rc == 0:
            from research.nnnc_switches.judge import judge_one

            s = judge_one(json.loads(PJ.PREREG.read_text()), "st_mofn48_c2", "st_mofn48")
            (PJ.HERE / "secondary.json").write_text(json.dumps(s, indent=1, default=str))
            print("secondary vs st_mofn48:", s["verdict"], s.get("arm_pp"), s.get("ctl_pp"),
                  s.get("early_gain_pp"), s.get("late_gain_pp"), (s.get("paired") or {}).get("verdict"))
        raise SystemExit(rc)
    PJ.PREREG, PJ.HERE = HERE / "preregistration.json", HERE
    raise SystemExit(PJ.main())
