# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNMT_LR_RI on the NNMT defaults with the v1 transition stand-in (inputs from the host dataframe,
saved_data/transition/): the class the mt_lr_ri_dtr check record ran (44.786; 12/6854 flags differed from
the tables through fast_diff residue and the 64-candle warm-up). Same weights."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_LR_RI import NNMT_LR_RI


class NNMT_LR_RI_TRV1(NNMT_LR_RI):
    TR_STANDIN_ROOT = None
