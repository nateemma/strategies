"""Composite sizer study: the HGB sizer's instrument checks (research/hgb_sizer/instrument.py) on
`cs_cmp`, against the composite tables. No P&L is read.

    PYTHONPATH=. ../../.venv/bin/python -m research.cmp_sizer.instrument
"""
from __future__ import annotations

from pathlib import Path

from research.cmp_sizer.signals import OUT
from research.hgb_sizer import instrument as HI

HI.OUT, HI.HERE = OUT, Path(__file__).resolve().parent

if __name__ == "__main__":
    raise SystemExit(HI.main())
