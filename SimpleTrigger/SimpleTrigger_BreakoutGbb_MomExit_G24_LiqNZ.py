# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Pair-relative liquidity guard on the pair's TRADED candles: signal candle quote volume above 5x the median of the non-zero-volume candles in the trailing 30 days (2880 bars). Pre-registered in research/tb_model/preregistration_liqnz.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_BreakoutGbb_MomExit_G24 import SimpleTrigger_BreakoutGbb_MomExit_G24
from utils.LiquidityGuardMixin import LiquidityGuardMixin
from utils.StakeScaleMixin import StakeScaleMixin


class SimpleTrigger_BreakoutGbb_MomExit_G24_LiqNZ(LiquidityGuardMixin, StakeScaleMixin, SimpleTrigger_BreakoutGbb_MomExit_G24):
    ATTR_OVERRIDES = {**SimpleTrigger_BreakoutGbb_MomExit_G24.ATTR_OVERRIDES, "startup_candle_count": 4096}
    LIQUIDITY_GUARD = {"window": 2880, "min_ratio": 5.0, "nonzero": True}
