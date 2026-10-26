# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""040 model variant (research R6, preregistration_models.json): current network on raw8, at ARO's operating point
(guards open, threshold 0.80). Same label, task weights, seq_len and exits as NNMT_MLX_Conv."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from _nnmt_operating_point import OperatingPointMixin
from _nnmt_inputs import NNMTInputSetMixin
from NNMT_MLX_Conv import NNMT_MLX_Conv


class NNMT_Raw(NNMTInputSetMixin, OperatingPointMixin, NNMT_MLX_Conv):
    INPUT_SET = "raw"
