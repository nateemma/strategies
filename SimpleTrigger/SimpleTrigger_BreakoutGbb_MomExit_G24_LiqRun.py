# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""Signal-rank liquidity guard: keep a signal only where its candle's quote volume beats the 2/3 quantile of the pair's previous 160 traded signal candles (~3 weeks), one decision per signal run, taken at the run's first traded row. Pre-registered in research/tb_model/preregistration_liqsr2.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from SimpleTrigger.SimpleTrigger_BreakoutGbb_MomExit_G24 import SimpleTrigger_BreakoutGbb_MomExit_G24
from utils.LiquidityGuardMixin import LiquidityGuardMixin
from utils.StakeScaleMixin import StakeScaleMixin


class SimpleTrigger_BreakoutGbb_MomExit_G24_LiqRun(LiquidityGuardMixin, StakeScaleMixin, SimpleTrigger_BreakoutGbb_MomExit_G24):
    ATTR_OVERRIDES = {**SimpleTrigger_BreakoutGbb_MomExit_G24.ATTR_OVERRIDES, "startup_candle_count": 4096}
    LIQUIDITY_GUARD = {"signals": 160, "quantile": 2 / 3, "runs": True}
