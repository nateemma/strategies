"""Composite sizer study: the per-fold agree threshold for `cs_cmp` -- the HGB sizer's pick and verify
(research/hgb_sizer/pick.py: cs_aro's rule) on plan_cs_cmp.json and the composite tables. The plan
is passed explicitly: hgb_sizer.pick binds plan_path=PLAN as a default at definition time.

    PYTHONPATH=. ../../.venv/bin/python -m research.cmp_sizer.pick --fold f00 [--verify]
"""
from __future__ import annotations

import argparse
import json

from research.cmp_sizer.signals import OUT
from research.hgb_sizer import pick as HP

PLAN = HP.CP.FOLDS / "plan_cs_cmp.json"
HP.OUT = OUT                     # read at call time inside pick()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="exactly one fold id")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    if a.verify:
        return 0 if HP.verify(a.fold, PLAN) else 1
    t, extra = HP.pick(a.fold, PLAN)
    print(f"cs_cmp {a.fold}: threshold {HP.CP.store(PLAN, a.fold, t, extra)} {json.dumps(extra)}")
    return 0 if not extra["tail_leak_pairs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
