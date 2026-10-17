# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""037 M-of-N vote `mofn_4of8` as a SimpleTrigger arm (research/mofn_trigger/): reversion family exits on
the ARM_D_FAMILY preset, the same as the gbb comparator `st_fam_gbb` (SimpleTrigger_FamGbb)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


class SimpleTrigger_MofN4of8(SimpleTriggerStrategy):
    TRIGGER_NAME = "mofn_4of8"
    PRESET = "ARM_D_FAMILY"
