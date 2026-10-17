"""ARO volume-filter retrain: the pre-registered verdict for `cs_arov` -- the HGB sizer's judge (cs_aro's
bar vs w_aro and wf_aro_g24, plus judge_one vs cs_aro) on this folder's preregistration.json.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.aro_volume.judge
"""
from __future__ import annotations

from pathlib import Path

from research.hgb_sizer import judge as HJ

HJ.HERE = Path(__file__).resolve().parent
HJ.PREREG = HJ.HERE / "preregistration.json"

if __name__ == "__main__":
    raise SystemExit(HJ.main())
