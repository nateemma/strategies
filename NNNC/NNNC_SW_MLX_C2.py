# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNNC switch experiment (2026-10-12): 028's ATR take-profit (C2) for the reversion family.

`NNNC_MLX` on its own borrowed per-fold weights (`NNNC_WF_fNN`) and per-fold thresholds, with ONE
framework switch declared -- no other change. Pre-registered in research/nnnc_switches/.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from NNNC_MLX import NNNC_MLX


class NNNC_SW_MLX_C2(NNNC_MLX):
    # Weights: `NNNC_WF_fNN`, found by model key (030; was BorrowedModelMixin's MODEL_SOURCE).
    FAMILY_EXIT_OVERRIDES = {"reversion": {"take_profit_atr_mult": 2.8970711921574543}}
