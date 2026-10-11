# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""`gbb` on the ARM_D_FAMILY preset: it runs the reversion family's exits (feature 024 US1).

SC-001: must reproduce `st_gbb` trade-for-trade in 15/15 folds."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


class SimpleTrigger_FamGbb(SimpleTriggerStrategy):
    TRIGGER_NAME = "gbb"
    PRESET = "ARM_D_FAMILY"
