# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Reversion tuning step 1: take_profit = 10.0, on BASE_ARM st_fam_combo (feature 025 US2)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_FamCombined import SimpleTrigger_FamCombined


class SimpleTrigger_Tune1TpOff(SimpleTrigger_FamCombined):
    FAMILY_EXIT_OVERRIDES = {"reversion": {"take_profit": 10.0}}
