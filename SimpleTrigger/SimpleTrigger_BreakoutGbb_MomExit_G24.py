# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""SimpleTrigger grace study (research/st_grace, 2026-10-13): the momentum reference st_bgo_mx with NNNC's adopted
24 h grace window (ed6049b) in place of 6.5 h, same -27% depth. Pre-registered arm."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_BreakoutGbb_MomExit import SimpleTrigger_BreakoutGbb_MomExit


class SimpleTrigger_BreakoutGbb_MomExit_G24(SimpleTrigger_BreakoutGbb_MomExit):
    SELL_OVERRIDES = {**SimpleTrigger_BreakoutGbb_MomExit.SELL_OVERRIDES, "stoploss_grace_hours": 24.0}
