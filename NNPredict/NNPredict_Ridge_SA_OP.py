# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""039 operating-point twin of NNPredict_Ridge_SA (NNPredict/_operating_point.py): guards open + percentile
magnitude floor; reuses NNPredict_Ridge_SA's trained weights (post-model settings only)."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _operating_point import OperatingPointMixin
from NNPredict_Ridge_SA import NNPredict_Ridge_SA


class NNPredict_Ridge_SA_OP(OperatingPointMixin, NNPredict_Ridge_SA):
    pass
