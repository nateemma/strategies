# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""`failed_breakout` on arm D's (reversion) exits, the default ARM_D_OPEN preset. A candidate
reversion trigger alongside gbb (023 Addendum 7)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


class SimpleTrigger_FailedBreakout(SimpleTriggerStrategy):
    TRIGGER_NAME = "failed_breakout"
