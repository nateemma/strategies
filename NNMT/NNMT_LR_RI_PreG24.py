# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNMT_LR_RI with NNMTStrategy's pre-040-adoption stop-loss grace (2.6 h, _pre_g24.py): the class its 040
records ran. Same weights."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _pre_g24 import PreG24GraceMixin
from NNMT_LR_RI import NNMT_LR_RI


class NNMT_LR_RI_PreG24(PreG24GraceMixin, NNMT_LR_RI):
    pass
