# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""`NNNC_CS_ARO_ALL3K` (4096-bar startup) with the signal-rank liquidity guard: keep a signal only where its candle's quote volume beats the 2/3 quantile of the pair's previous 20 traded signal candles (~3 weeks), one decision per signal run, taken at the run's first traded row. Pre-registered in research/tb_model/preregistration_liqsr2.json."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from NNNC_CS_ARO_ALL3K import NNNC_CS_ARO_ALL3K
from utils.LiquidityGuardMixin import LiquidityGuardMixin


class NNNC_CS_ARO_ALL3K_LIQRUN(LiquidityGuardMixin, NNNC_CS_ARO_ALL3K):
    startup_candle_count = 4096
    # 030: warm-up is a training input; these weights were trained with 64 startup candles.
    MODEL_TRAINED_WITH = {"startup_candle_count": 64}
    LIQUIDITY_GUARD = {"signals": 20, "quantile": 2 / 3, "runs": True}
