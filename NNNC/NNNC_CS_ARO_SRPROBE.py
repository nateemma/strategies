# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Signal-rank liquidity guard in OBSERVE mode (filters nothing) on `NNNC_CS_ARO` (4096-bar startup): liveness probe and signal-history dump; must reproduce cs_aro_s4k trade for trade. For research/tb_model/preregistration_liqsr.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from NNNC_CS_ARO import NNNC_CS_ARO
from utils.LiquidityGuardMixin import LiquidityGuardMixin


class NNNC_CS_ARO_SRPROBE(LiquidityGuardMixin, NNNC_CS_ARO):
    startup_candle_count = 4096
    # 030: warm-up is a training input; these weights were trained with 64 startup candles.
    MODEL_TRAINED_WITH = {"startup_candle_count": 64}
    LIQUIDITY_GUARD = {"signals": 20, "quantile": 0.5, "observe": True}
