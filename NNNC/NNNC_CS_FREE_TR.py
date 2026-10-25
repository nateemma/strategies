# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""cs_free + the 040 tr_scale exposure scaler, with NNNC_CS_ARO_TR's settings (preregistration_cs_free.json)."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _transition_signal import TrScaleMixin
from NNNC_CS_FREE import NNNC_CS_FREE


class NNNC_CS_FREE_TR(TrScaleMixin, NNNC_CS_FREE):
    TR_UP_CELL = "up_pair_96"
    TR_DOWN_CELL = "down_pair_672"
    TR_SOURCE = "standin"
    TR_STANDIN_ROOT = "transition_v2"
