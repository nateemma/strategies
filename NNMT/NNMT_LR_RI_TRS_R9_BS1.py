# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""040 rounding amendment arm tr_scale_mt_r9_bs1 (research/nnmt/preregistration_transition_round.json): NNMT_LR_RI on the NNMT defaults (bootstrap fit) on the ROUNDED
transition stand-in (saved_data/transition_r9_bs1/, inputs rounded to 9 decimals), scored in-strategy. Same weights."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_LR_RI import NNMT_LR_RI


class NNMT_LR_RI_TRS_R9_BS1(NNMT_LR_RI):
    TR_STANDIN_ROOT = "transition_r9_bs1"
