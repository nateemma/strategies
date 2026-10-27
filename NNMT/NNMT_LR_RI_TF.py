# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""TensorFlow twin of NNMT_LR_RI (= NNMT_MLX): the same strategy, its multi-task LSTM_RAW on TensorFlow
(NNMT/NNMTClassifierTF.py), for hosts without Apple's MLX. Like its parent it has no .json."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from NNMT.NNMTClassifierTF import TFMultiTaskClassifierMixin  # noqa: E402
from NNMT_LR_RI import NNMT_LR_RI  # noqa: E402


class NNMT_LR_RI_TF(TFMultiTaskClassifierMixin, NNMT_LR_RI):
    pass
