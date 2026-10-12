# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Null check for MOMENTUM_PREEMPT (026 addendum 2): reversion disabled, so no preempt can fire and
every +late signal must be refused -- must reproduce st_fam_combo_mom exactly."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_FamCombinedMomOnly import SimpleTrigger_FamCombinedMomOnly


class SimpleTrigger_PreemptMomOnly(SimpleTrigger_FamCombinedMomOnly):
    MOMENTUM_PREEMPT = True
