# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""`breakout_gm` (price breakout AND guard_metric > 0.2) on the momentum exits: no take-profit,
unclog_12h deferred to 24h (023 Addendum 4)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from NNNC._hold_window import HoldWindowMixin
from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


class SimpleTrigger_BreakoutGm_MomExit(HoldWindowMixin, SimpleTriggerStrategy):
    TRIGGER_NAME = "breakout_gm"
    SELL_OVERRIDES = {"cexit_take_profit": 10.0}
    HOLD_WINDOW_HOURS = 24.0
