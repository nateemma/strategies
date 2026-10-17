"""Cross-model sizer study: thresholds for `cs_<m>_mix` -- the HGB sizer's pick and verify on
plan_cs_<m>_mix.json, the plan passed explicitly (hgb_sizer.pick binds its plan as a default argument).
The plain `cs_<m>` arms use research/control_sizer/pick.py --plan plan_cs_<m>.json.

    PYTHONPATH=. ../../.venv/bin/python -m research.xmodel_sizer.pick --model MLX --fold f00 [--verify]
"""
from __future__ import annotations

import argparse
import json

from research.hgb_sizer import pick as HP
from research.xmodel_sizer.knots import out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("GBR", "MLX"))
    ap.add_argument("--fold", required=True)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    plan = HP.CP.FOLDS / f"plan_cs_{a.model.lower()}_mix.json"
    HP.OUT = out_dir(a.model)                     # read at call time inside pick()
    if a.verify:
        return 0 if HP.verify(a.fold, plan) else 1
    t, extra = HP.pick(a.fold, plan)
    print(f"cs_{a.model.lower()}_mix {a.fold}: threshold {HP.CP.store(plan, a.fold, t, extra)} {json.dumps(extra)}")
    return 0 if not extra["tail_leak_pairs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
