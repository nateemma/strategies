"""Composite sizer study: the per-fold agree threshold for `cs_cmp` -- the HGB sizer's pick
(research/hgb_sizer/pick.py: cs_aro's rule, verify) on plan_cs_cmp.json and the composite tables.

    PYTHONPATH=. ../../.venv/bin/python -m research.cmp_sizer.pick --fold f00 [--verify]
"""
from __future__ import annotations

from research.cmp_sizer.signals import OUT
from research.hgb_sizer import pick as HP

HP.PLAN, HP.OUT = HP.CP.FOLDS / "plan_cs_cmp.json", OUT

if __name__ == "__main__":
    raise SystemExit(HP.main())
