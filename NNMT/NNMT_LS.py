# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""040 architecture amendment (preregistration_arch.json): LSTM_STACK (shared LSTM + an LSTM per task
head; option 3) on the 24 indicators, at ARO's operating point (guards open, threshold 0.80). Differs from
NNMT_LR (option 1: shared LSTM + MLP head per task) only in the classifier."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from _nnmt_operating_point import OperatingPointMixin
from NNMT.NNMTClassifierMLX import ClassifierTypeMLX
from NNMT_MLX_Conv import NNMT_MLX_Conv


class NNMT_LS(OperatingPointMixin, NNMT_MLX_Conv):
    classifier_type = ClassifierTypeMLX.LSTM_STACK
