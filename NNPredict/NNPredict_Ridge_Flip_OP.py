# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""039 threshold amendment, option 1: NNPredict_Ridge_OP trading the sign-flipped prediction
(NNPredict/_flip.py). Reuses NNPredict_Ridge's trained weights."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _flip import FlipSignMixin
from NNPredict_Ridge_OP import NNPredict_Ridge_OP


class NNPredict_Ridge_Flip_OP(FlipSignMixin, NNPredict_Ridge_OP):
    pass
