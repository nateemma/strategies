"""Mixed sizer study: the pre-registered verdict for `cs_mix` -- the HGB sizer's judge on this folder's
preregistration.json and instrument.json; plus judge_one(cs_mix, cs_cmp), descriptive, in vs_cs_cmp.json.

    PYTHONPATH=. ../../.venv/bin/python -m research.mix_sizer.judge
"""
from __future__ import annotations

import json
from pathlib import Path

from research.hgb_sizer import judge as HJ

HJ.HERE = Path(__file__).resolve().parent
HJ.PREREG = HJ.HERE / "preregistration.json"

if __name__ == "__main__":
    rc = HJ.main()
    if rc == 0 and json.loads((HJ.HERE / "instrument.json").read_text()).get("pass"):
        from research.nnnc_switches.judge import judge_one

        r = judge_one(json.loads(HJ.PREREG.read_text()), "cs_mix", "cs_cmp")
        (HJ.HERE / "vs_cs_cmp.json").write_text(json.dumps(r, indent=1, default=str))
        print("vs cs_cmp (descriptive):", r["verdict"], r.get("arm_pp"), r.get("ctl_pp"),
              r.get("early_gain_pp"), r.get("late_gain_pp"))
    raise SystemExit(rc)
