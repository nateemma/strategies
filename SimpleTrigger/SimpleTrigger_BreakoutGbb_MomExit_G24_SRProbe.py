# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Signal-rank liquidity guard in OBSERVE mode (filters nothing): liveness probe and signal-history dump for research/tb_model/preregistration_liqsr.json; must reproduce st_g24_mom_s4k trade for trade."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_BreakoutGbb_MomExit_G24 import SimpleTrigger_BreakoutGbb_MomExit_G24
from utils.LiquidityGuardMixin import LiquidityGuardMixin
from utils.StakeScaleMixin import StakeScaleMixin


class SimpleTrigger_BreakoutGbb_MomExit_G24_SRProbe(LiquidityGuardMixin, StakeScaleMixin, SimpleTrigger_BreakoutGbb_MomExit_G24):
    ATTR_OVERRIDES = {**SimpleTrigger_BreakoutGbb_MomExit_G24.ATTR_OVERRIDES, "startup_candle_count": 4096}
    LIQUIDITY_GUARD = {"signals": 20, "quantile": 0.5, "observe": True}
