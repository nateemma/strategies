"""Mixed sizer study: the HGB sizer's instrument checks on `cs_mix`, against the mixed-score tables.

    PYTHONPATH=. ../../.venv/bin/python -m research.mix_sizer.instrument
"""
from __future__ import annotations

from pathlib import Path

from research.hgb_sizer import instrument as HI
from research.mix_sizer.knots import OUT

HI.OUT, HI.HERE = OUT, Path(__file__).resolve().parent

if __name__ == "__main__":
    raise SystemExit(HI.main())
