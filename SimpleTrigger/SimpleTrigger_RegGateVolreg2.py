# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""The volreg gate tightened to the stacked-feasibility p50 (026 addendum: second gate on "on" trades).

Rule from specs/026-reversion-regime-gate/feasibility_stack.json; it subsumes the 1.0332 cut."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_RegGateVolreg import SimpleTrigger_RegGateVolreg


class SimpleTrigger_RegGateVolreg2(SimpleTrigger_RegGateVolreg):
    REVERSION_REGIME_GATE = {
        "instrument": "volreg",
        "threshold": 1.3658023280926226,
        "off_when": "lt",
    }
