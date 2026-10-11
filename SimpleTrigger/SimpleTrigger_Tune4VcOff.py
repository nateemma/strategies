# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Reversion tuning step 4: volume-confirmation stop off (feature 025 US2).

Built on st_fam_combo; steps 1-3 not adopted."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_FamCombined import SimpleTrigger_FamCombined


class SimpleTrigger_Tune4VcOff(SimpleTrigger_FamCombined):
    FAMILY_EXIT_OVERRIDES = {"reversion": {"use_volume_confirmation_stoploss": False}}
