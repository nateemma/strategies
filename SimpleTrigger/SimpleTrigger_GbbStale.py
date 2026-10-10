# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""`gbb_stale` through arm D's framework (ARM_D_OPEN), no model -- the all-trigger sweep."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


class SimpleTrigger_GbbStale(SimpleTriggerStrategy):
    TRIGGER_NAME = "gbb_stale"
