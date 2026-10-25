# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""v2 parity arm tr_scale_mt_v2: NNMT_LR_RI on the NNMT defaults with the transition stand-in v2 (inputs from the history store's OHLCV,
saved_data/transition_v2/, unrounded = the tested stand-in). Must reproduce tr_scale_mt exactly. Same weights."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_LR_RI import NNMT_LR_RI


class NNMT_LR_RI_TRS_V2(NNMT_LR_RI):
    TR_STANDIN_ROOT = "transition_v2"
