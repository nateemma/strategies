# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNNC switch experiment (2026-10-12): 026's volreg reversion gate (st_book_rev2's threshold), 4096-bar startup.

`NNNC_MLX` on its own borrowed per-fold weights (`NNNC_WF_fNN`) and per-fold thresholds, with ONE
framework switch declared -- no other change. Pre-registered in research/nnnc_switches/.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from BorrowedModelMixin import BorrowedModelMixin
from NNNC_MLX import NNNC_MLX


class NNNC_SW_MLX_Gate(BorrowedModelMixin, NNNC_MLX):
    MODEL_SOURCE = "NNNC_WF"
    startup_candle_count = 4096
    REVERSION_REGIME_GATE = {"instrument": "volreg", "threshold": 1.3658023280926226, "off_when": "lt"}
