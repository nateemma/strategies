# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Momentum-exit cheap check: SimpleTrigger_BreakoutBb with no take-profit and unclog_12h deferred to 24h.

Tests whether momentum triggers want a longer cycle than the shared ARM_D_OPEN exit stack allows.
Pair against st_bbo (full) and st_bbo__take_profit (take-profit off only)."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from NNNC._hold_window import HoldWindowMixin
from SimpleTrigger.SimpleTrigger_BreakoutBb import SimpleTrigger_BreakoutBb


class SimpleTrigger_BreakoutBb_MomExit(HoldWindowMixin, SimpleTrigger_BreakoutBb):
    SELL_OVERRIDES = {"cexit_take_profit": 10.0}
    HOLD_WINDOW_HOURS = 24.0
