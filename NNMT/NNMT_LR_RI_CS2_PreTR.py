# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNMT_LR_RI_CS2 with NNMTStrategy's pre-040-adoption transition setting (tr_scale off (its own reversion tags kept: the sizer reads them); _pre_g24.py): the class its 040
records ran. Same weights."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _pre_g24 import PreTrScaleOffMixin
from NNMT_LR_RI_CS2 import NNMT_LR_RI_CS2


class NNMT_LR_RI_CS2_PreTR(PreTrScaleOffMixin, NNMT_LR_RI_CS2):
    pass
