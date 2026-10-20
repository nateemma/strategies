# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""040 model variant (research R6, preregistration_models.json): LSTM_RAW on raw8, at ARO's operating point
(guards open, threshold 0.80). Same label, task weights, seq_len and exits as NNMT_MLX."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from _nnmt_operating_point import OperatingPointMixin
from _nnmt_inputs import NNMTInputSetMixin
from NNMT.NNMTClassifierMLX import ClassifierTypeMLX
from NNMT_MLX import NNMT_MLX


class NNMT_LR_Raw(NNMTInputSetMixin, OperatingPointMixin, NNMT_MLX):
    classifier_type = ClassifierTypeMLX.LSTM_RAW
    INPUT_SET = "raw"
