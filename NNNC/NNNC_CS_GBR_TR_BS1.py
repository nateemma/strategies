# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""040 follow-up arm tr_scale_gbr_bs1 (research/nnmt/preregistration_trscale_ext.json): tr_scale on cs_gbr (GBR's
width-book sizer). Bootstrap stand-in tables (seed rule)."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _transition_signal import TABLES, TrScaleMixin
from NNNC_CS_GBR import NNNC_CS_GBR


class NNNC_CS_GBR_TR_BS1(TrScaleMixin, NNNC_CS_GBR):
    TR_UP_CELL = "up_pair_96"
    TR_DOWN_CELL = "down_pair_672"
    TR_TABLES = TABLES.parent / "nnmt_transition_bs1"
