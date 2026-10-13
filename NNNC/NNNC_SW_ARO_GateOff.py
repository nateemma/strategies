# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNNC switch experiment (2026-10-12): the regime gate's control -- 4096-bar startup, gate OFF.

`NNNC_A_Raw_Open` on its own borrowed per-fold weights (`NNNC_ARO_fNN`) and per-fold thresholds, with ONE
framework switch declared -- no other change. Pre-registered in research/nnnc_switches/.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from NNNC_A_Raw_Open import NNNC_A_Raw_Open


class NNNC_SW_ARO_GateOff(NNNC_A_Raw_Open):
    # Weights: `NNNC_ARO_fNN`, found by model key (030; was BorrowedModelMixin's MODEL_SOURCE).
    startup_candle_count = 4096
    # 030: warm-up is a training input; these weights were trained with 64 startup candles.
    MODEL_TRAINED_WITH = {"startup_candle_count": 64}
