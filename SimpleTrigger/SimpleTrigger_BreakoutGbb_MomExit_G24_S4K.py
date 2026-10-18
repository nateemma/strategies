# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Warm-up null for the liquidity arms: the momentum book (st_g24_mom) with a 4096-bar startup and no guard. Pre-registered in research/tb_model/preregistration_liq.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_BreakoutGbb_MomExit_G24 import SimpleTrigger_BreakoutGbb_MomExit_G24
from utils.LiquidityGuardMixin import LiquidityGuardMixin
from utils.StakeScaleMixin import StakeScaleMixin


class SimpleTrigger_BreakoutGbb_MomExit_G24_S4K(LiquidityGuardMixin, StakeScaleMixin, SimpleTrigger_BreakoutGbb_MomExit_G24):
    ATTR_OVERRIDES = {**SimpleTrigger_BreakoutGbb_MomExit_G24.ATTR_OVERRIDES, "startup_candle_count": 4096}

