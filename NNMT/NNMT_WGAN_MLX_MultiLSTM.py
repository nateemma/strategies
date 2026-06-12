# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNMT_WGAN_MLX_MultiLSTM - Subclass of NNMT_WGAN using MLX MultiLSTM classifier.

The MLX classifier factory is provided by MLXMultiTaskClassifierMixin.
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMT_WGAN import NNMT_WGAN  # noqa: E402
from NNMT.NNMTClassifierMLX import ClassifierTypeMLX, MLXMultiTaskClassifierMixin


class NNMT_WGAN_MLX_MultiLSTM(MLXMultiTaskClassifierMixin, NNMT_WGAN):

    classifier_type = ClassifierTypeMLX.Multi_LSTM
