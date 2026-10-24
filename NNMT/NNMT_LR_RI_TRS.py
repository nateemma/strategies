# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""040 follow-up arm tr_scale_mt (research/nnmt/preregistration_trscale_ext.json): tr_scale on NNMT_LR_RI at
the NNMT defaults (24 h grace). Entries tagged reversion:model so the scaler reads the book's mask."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "NNNC"))

from _stake_weight import StakeWeightMixin
from _transition_signal import TABLES, TrScaleMixin
from NNMT_LR_RI import NNMT_LR_RI


class NNMT_LR_RI_TRS(TrScaleMixin, StakeWeightMixin, NNMT_LR_RI):
    MODEL_ENTRY_FAMILY = "reversion"
    TR_UP_CELL = "up_pair_96"
    TR_DOWN_CELL = "down_pair_672"

