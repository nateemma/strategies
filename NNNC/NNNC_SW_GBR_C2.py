# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNNC switch experiment (2026-10-12): 028's ATR take-profit (C2) for the reversion family.

`NNNC_GbbRaw_Open` on its own borrowed per-fold weights (`NNNC_GBR_fNN`) and per-fold thresholds, with ONE
framework switch declared -- no other change. Pre-registered in research/nnnc_switches/.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from BorrowedModelMixin import BorrowedModelMixin
from NNNC_GbbRaw_Open import NNNC_GbbRaw_Open


class NNNC_SW_GBR_C2(BorrowedModelMixin, NNNC_GbbRaw_Open):
    MODEL_SOURCE = "NNNC_GBR"
    FAMILY_EXIT_OVERRIDES = {"reversion": {"take_profit_atr_mult": 2.8970711921574543}}
