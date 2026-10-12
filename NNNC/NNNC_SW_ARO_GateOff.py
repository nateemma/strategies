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

from BorrowedModelMixin import BorrowedModelMixin
from NNNC_A_Raw_Open import NNNC_A_Raw_Open


class NNNC_SW_ARO_GateOff(BorrowedModelMixin, NNNC_A_Raw_Open):
    MODEL_SOURCE = "NNNC_ARO"
    startup_candle_count = 4096
