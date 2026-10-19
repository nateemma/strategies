# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""039 (research R4): NNPredict_LSTM_Raw on the `raw_ret` input set (NNPredict/_inputs.py)."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNPredict_LSTM_Raw import NNPredict_LSTM_Raw


class NNPredict_LSTM_RawRet(NNPredict_LSTM_Raw):
    INPUT_SET = "raw_ret"
