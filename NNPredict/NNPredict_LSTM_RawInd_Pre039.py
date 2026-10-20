# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNPredict_LSTM_RawInd with NNPredictStrategy's pre-039 post-model settings (_pre039.py): the class its 039 record
ran. Same weights."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _pre039 import Pre039DefaultsMixin
from NNPredict_LSTM_RawInd import NNPredict_LSTM_RawInd


class NNPredict_LSTM_RawInd_Pre039(Pre039DefaultsMixin, NNPredict_LSTM_RawInd):
    pass
