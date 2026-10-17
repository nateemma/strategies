# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""037 signal pair `pair_kst_zscore` as a SimpleTrigger arm (research/pair_triggers/): reversion family exits on
the ARM_D_FAMILY preset, the same as the gbb comparator `st_fam_gbb` (SimpleTrigger_FamGbb)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


class SimpleTrigger_PT_KstZscore(SimpleTriggerStrategy):
    TRIGGER_NAME = "pair_kst_zscore"
    PRESET = "ARM_D_FAMILY"
