# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""TensorFlow twin of NNNC_A_Raw_Open (ARO): the same strategy, its LSTM_RAW model on TensorFlow
(NNNC/NNNClassifierTF.py), for hosts without Apple's MLX. Own model key: it trains its own weights, or
loads weights converted from ARO's (scripts/convert_mlx_to_tf.py). Its .json is a copy of the parent's."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from NNNClassifierTF import TFClassifierMixin  # noqa: E402
from NNNC_A_Raw_Open import NNNC_A_Raw_Open  # noqa: E402


class NNNC_A_Raw_Open_TF(TFClassifierMixin, NNNC_A_Raw_Open):
    pass
