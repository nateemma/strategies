"""mofn_4of8 as a trigger: the pre-registered verdicts (preregistration.json). Same rules and code as
research/pair_triggers/judge.py, pointed at this pre-registration.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.mofn_trigger.judge
"""
from __future__ import annotations

from pathlib import Path

from research.pair_triggers import judge as PJ

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    PJ.PREREG, PJ.HERE = HERE / "preregistration.json", HERE
    raise SystemExit(PJ.main())
