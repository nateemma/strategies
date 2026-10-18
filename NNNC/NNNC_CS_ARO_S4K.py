# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""`NNNC_CS_ARO` with a 4096-bar startup and no guard: the warm-up null the liquidity guard needs. Pre-registered in research/tb_model/preregistration_liqnz.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from NNNC_CS_ARO import NNNC_CS_ARO
from utils.LiquidityGuardMixin import LiquidityGuardMixin


class NNNC_CS_ARO_S4K(LiquidityGuardMixin, NNNC_CS_ARO):
    startup_candle_count = 4096
    # 030: warm-up is a training input; these weights were trained with 64 startup candles.
    MODEL_TRAINED_WITH = {"startup_candle_count": 64}
