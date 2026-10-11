# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Reversion entries off when the cross-pair volatility regime is below its early-chosen p33.

Rule from specs/026-reversion-regime-gate/feasibility.json (feature 026)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_FamCombined import SimpleTrigger_FamCombined


class SimpleTrigger_RegGateVolreg(SimpleTrigger_FamCombined):
    ATTR_OVERRIDES = {"startup_candle_count": 4096}
    REVERSION_REGIME_GATE = {
        "instrument": "volreg",
        "threshold": 1.033196702052918,
        "off_when": "lt",
    }
