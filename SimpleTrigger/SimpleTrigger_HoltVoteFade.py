# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""038 vote C mirror `holt_vote_fade` as a SimpleTrigger arm (research/holt_trigger/): 3 of 4 damped-Holt
windows forecast < -30 bp over 12 h; the long-only fade (buy after the trend models call a fall).
Reversion family, so the gbb comparator's exits: everything inherited from st_fam_gbb
(SimpleTrigger_FamGbb, ARM_D_FAMILY); only the trigger differs."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_FamGbb import SimpleTrigger_FamGbb


class SimpleTrigger_HoltVoteFade(SimpleTrigger_FamGbb):
    TRIGGER_NAME = "holt_vote_fade"
