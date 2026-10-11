# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""`breakout_gbb_nofresh` on the momentum exits (no take-profit, unclog_12h deferred to 24h).
Pair against st_bgo_mx: tests Addendum 6's prediction that fresh/held breakouts drag breakout_gbb
down (023 Addendum 7)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from NNNC._hold_window import HoldWindowMixin
from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


class SimpleTrigger_BreakoutGbbNoFresh_MomExit(HoldWindowMixin, SimpleTriggerStrategy):
    TRIGGER_NAME = "breakout_gbb_nofresh"
    SELL_OVERRIDES = {"cexit_take_profit": 10.0}
    HOLD_WINDOW_HOURS = 24.0
