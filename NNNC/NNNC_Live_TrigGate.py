# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""029 LIVENESS ARM (T028): trigger AND model: enter only on gbb bars the model accepts.

Arm D (`NNNC_A_Raw_Open`) on its own borrowed per-fold weights, with ONE framework switch declared
-- no other change. Compared on f00 with arm D's after-change f00.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from BorrowedModelMixin import BorrowedModelMixin
from NNNC_A_Raw_Open import NNNC_A_Raw_Open


class NNNC_Live_TrigGate(BorrowedModelMixin, NNNC_A_Raw_Open):
    ENTRY_MODE = "trigger_and_model"
    TRIGGER_NAME = "gbb"
