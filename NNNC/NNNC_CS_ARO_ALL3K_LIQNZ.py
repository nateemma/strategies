# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""`NNNC_CS_ARO_ALL3K` (4096-bar startup) with the non-zero-median liquidity guard (5x, 2880 bars). Pre-registered in research/tb_model/preregistration_liqnz.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from NNNC_CS_ARO_ALL3K import NNNC_CS_ARO_ALL3K
from utils.LiquidityGuardMixin import LiquidityGuardMixin


class NNNC_CS_ARO_ALL3K_LIQNZ(LiquidityGuardMixin, NNNC_CS_ARO_ALL3K):
    startup_candle_count = 4096
    # 030: warm-up is a training input; these weights were trained with 64 startup candles.
    MODEL_TRAINED_WITH = {"startup_candle_count": 64}
    LIQUIDITY_GUARD = {"window": 2880, "min_ratio": 5.0, "nonzero": True}
