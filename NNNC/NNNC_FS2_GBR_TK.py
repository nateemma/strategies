# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNNC framework study, stage 2 (research/nnnc_framework/): exit:tp_atr_k = 6.0 on `NNNC_GbbRaw_Open`.

One lever changed, nothing else. Weights: the base's per-fold models, found by model key (030).
Pre-registered in research/nnnc_framework/preregistration.json.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from NNNC_GbbRaw_Open import NNNC_GbbRaw_Open


class NNNC_FS2_GBR_TK(NNNC_GbbRaw_Open):
    FAMILY_EXIT_OVERRIDES = {"reversion": {'take_profit_atr_mult': 6.0}}
