# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Combined reversion + momentum arm, both sides (feature 024 US2).

gbb (reversion exits) and breakout_gbb (momentum exits) share slots and one-position-per-pair;
overlap bars go to reversion (FAMILY_PRIORITY default) and are tagged +ov."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


class SimpleTrigger_FamCombined(SimpleTriggerStrategy):
    TRIGGERS = ("gbb", "breakout_gbb")
    PRESET = "ARM_D_FAMILY"
