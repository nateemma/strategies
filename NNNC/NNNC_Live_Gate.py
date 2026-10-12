# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""029 LIVENESS ARM (T028): 026's volreg reversion gate (st_book_rev2's threshold) on arm D's reversion-tagged entries.

Arm D (`NNNC_A_Raw_Open`) on its own borrowed per-fold weights, with ONE framework switch declared
-- no other change. Compared on f00 with NNNC_Live_GateOff (same 4096-bar startup, gate off).
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from BorrowedModelMixin import BorrowedModelMixin
from NNNC_A_Raw_Open import NNNC_A_Raw_Open


class NNNC_Live_Gate(BorrowedModelMixin, NNNC_A_Raw_Open):
    startup_candle_count = 4096
    REVERSION_REGIME_GATE = {"instrument": "volreg", "threshold": 1.3658023280926226, "off_when": "lt"}
