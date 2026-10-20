# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""039 threshold amendment, option 2b: NNPredict_Ridge at the operating point with HORIZON 24
(6 h: nnp_ridge_op's median hold is 3.5-7.25 h per fold) instead of 4 (1 h). New weights."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _operating_point import OperatingPointMixin
from NNPredict_Ridge import NNPredict_Ridge


class NNPredict_Ridge_H24_OP(OperatingPointMixin, NNPredict_Ridge):
    HORIZON = 24
