# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""`gbb` on the momentum exits (no take-profit, unclog_12h deferred to 24h) -- the fourth cell
of the trigger x exit-set grid used to price gbb/breakout_gbb overlap trades both ways
(023 Addendum 5)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from NNNC._hold_window import HoldWindowMixin
from SimpleTrigger.SimpleTrigger_Gbb import SimpleTrigger_Gbb


class SimpleTrigger_Gbb_MomExit(HoldWindowMixin, SimpleTrigger_Gbb):
    SELL_OVERRIDES = {"cexit_take_profit": 10.0}
    HOLD_WINDOW_HOURS = 24.0
