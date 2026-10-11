# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Rule B at its neutral setting: must reproduce st_fam_combo exactly (SC-001) (feature 025)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_FamCombined import SimpleTrigger_FamCombined


class SimpleTrigger_CoexWinOff(SimpleTrigger_FamCombined):
    MOMENTUM_RECENCY_BARS = 0
