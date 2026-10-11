# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Reversion tuning step 2: unclog_12h_hours = 24.0 (feature 025 US2).

Built on st_fam_combo; step 1 not adopted."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_FamCombined import SimpleTrigger_FamCombined


class SimpleTrigger_Tune2U24(SimpleTrigger_FamCombined):
    FAMILY_EXIT_OVERRIDES = {"reversion": {"unclog_12h_hours": 24.0}}
