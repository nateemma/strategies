# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNMT_MLX_MultiLSTM - Subclass of NNMTStrategy using MLX models
"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
from typing import Dict, Tuple
import mlx.core as mx

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMT_MLX import NNMT_MLX  # noqa: E402
from NNMT.NNMTClassifierMLX import ClassifierTypeMLX, create_classifier_mlx

# -----------


class NNMT_MLX_MultiLSTM(NNMT_MLX):

    classifier_type = ClassifierTypeMLX.Multi_LSTM


