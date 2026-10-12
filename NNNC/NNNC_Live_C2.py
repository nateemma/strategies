# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""029 LIVENESS ARM (T028): 028's ATR take-profit (C2) for the reversion family -- arm D's entries are reversion (gbb_path).

Arm D (`NNNC_A_Raw_Open`) on its own borrowed per-fold weights, with ONE framework switch declared
-- no other change. Compared on f00 with arm D's after-change f00.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from BorrowedModelMixin import BorrowedModelMixin
from NNNC_A_Raw_Open import NNNC_A_Raw_Open


class NNNC_Live_C2(BorrowedModelMixin, NNNC_A_Raw_Open):
    FAMILY_EXIT_OVERRIDES = {"reversion": {"take_profit_atr_mult": 2.8970711921574543}}
