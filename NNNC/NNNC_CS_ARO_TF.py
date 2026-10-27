# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""TensorFlow twin of NNNC_CS_ARO (the NNNC reference): the width book with ARO's p_buy as the stake
sizer, ARO's LSTM_RAW on TensorFlow (NNNC/NNNClassifierTF.py). Shares NNNC_A_Raw_Open_TF's model key (the
sizer and width book are post-model). Its .json is a copy of the parent's."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from NNNClassifierTF import TFClassifierMixin  # noqa: E402
from NNNC_CS_ARO import NNNC_CS_ARO  # noqa: E402


class NNNC_CS_ARO_TF(TFClassifierMixin, NNNC_CS_ARO):
    pass
