# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""039 operating-point twin of NNPredict_LSTM_RawInd (NNPredict/_operating_point.py): guards open + percentile
magnitude floor; reuses NNPredict_LSTM_RawInd's trained weights (post-model settings only)."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _operating_point import OperatingPointMixin
from NNPredict_LSTM_RawInd import NNPredict_LSTM_RawInd


class NNPredict_LSTM_RawInd_OP(OperatingPointMixin, NNPredict_LSTM_RawInd):
    pass
