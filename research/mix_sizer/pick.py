"""Mixed sizer study: the per-fold agree threshold for `cs_mix` -- the HGB sizer's pick and verify
(research/hgb_sizer/pick.py: cs_aro's rule) on plan_cs_mix.json, the plan passed explicitly.

    PYTHONPATH=. ../../.venv/bin/python -m research.mix_sizer.pick --fold f00 [--verify]
"""
from __future__ import annotations

import argparse
import json

from research.hgb_sizer import pick as HP
from research.mix_sizer.knots import OUT

PLAN = HP.CP.FOLDS / "plan_cs_mix.json"
HP.OUT = OUT                     # read at call time inside pick()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="exactly one fold id")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    if a.verify:
        return 0 if HP.verify(a.fold, PLAN) else 1
    t, extra = HP.pick(a.fold, PLAN)
    print(f"cs_mix {a.fold}: threshold {HP.CP.store(PLAN, a.fold, t, extra)} {json.dumps(extra)}")
    return 0 if not extra["tail_leak_pairs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
