# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""038 vote C `holt_vote_up` as a SimpleTrigger arm (research/holt_trigger/): 3 of 4 damped-Holt windows
forecast > +30 bp over 12 h. Momentum family, so the momentum reference's exits: everything inherited
from st_g24_mom (SimpleTrigger_BreakoutGbb_MomExit_G24); only the trigger differs."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_BreakoutGbb_MomExit_G24 import SimpleTrigger_BreakoutGbb_MomExit_G24


class SimpleTrigger_HoltVoteUp(SimpleTrigger_BreakoutGbb_MomExit_G24):
    TRIGGER_NAME = "holt_vote_up"
