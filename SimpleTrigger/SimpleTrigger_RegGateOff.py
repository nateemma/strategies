# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Regime gate disabled, default startup: must reproduce st_fam_combo (FR-006) (feature 026)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_FamCombined import SimpleTrigger_FamCombined


class SimpleTrigger_RegGateOff(SimpleTrigger_FamCombined):
    REVERSION_REGIME_GATE = None
