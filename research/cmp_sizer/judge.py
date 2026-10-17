"""Composite sizer study: the pre-registered verdict for `cs_cmp` -- the HGB sizer's judge
(research/hgb_sizer/judge.py) on this folder's preregistration.json and instrument.json.

    PYTHONPATH=. ../../.venv/bin/python -m research.cmp_sizer.judge
"""
from __future__ import annotations

from pathlib import Path

from research.hgb_sizer import judge as HJ

HJ.HERE = Path(__file__).resolve().parent
HJ.PREREG = HJ.HERE / "preregistration.json"

if __name__ == "__main__":
    raise SystemExit(HJ.main())
