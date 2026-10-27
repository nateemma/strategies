# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""040 model variant (research R6, preregistration_models.json): LSTM_RAW on the 24 indicators + raw8, at ARO's operating point
(guards open, threshold 0.80). Same label, task weights, seq_len and exits as NNMT_MLX_Conv."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from _nnmt_operating_point import OperatingPointMixin
from _nnmt_inputs import NNMTInputSetMixin
try:
    from NNMT.NNMTClassifierMLX import ClassifierTypeMLX
except ModuleNotFoundError as e:  # no MLX on this host: stay importable for the _TF twins
    if not (e.name or "").startswith("mlx"):
        raise
    from utils.MLXAbsent import ClassifierTypeMLX
from NNMT_MLX_Conv import NNMT_MLX_Conv


class NNMT_LR_RI(NNMTInputSetMixin, OperatingPointMixin, NNMT_MLX_Conv):
    classifier_type = ClassifierTypeMLX.LSTM_RAW
    INPUT_SET = "raw_ind"
