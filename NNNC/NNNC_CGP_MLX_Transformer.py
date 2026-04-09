# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
NNNC_CGP - Subclass of NNNCStrategy using CTAB-GAN+ for high-fidelity augmentation
MLX variants use Apple's native metal layers. Should be much faster
"""

import sys
from pathlib import Path
import os

import mlx.core as mx

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_CGP_MLX import NNNC_CGP_MLX
from NNNClassifierMLX import ClassifierTypeMLX, create_classifier_mlx
from ClassifierKeras import ClassifierKeras


class NNNC_CGP_MLX_Transformer(NNNC_CGP_MLX):

    # default is LSTM type. Override get_classifier_type() in subclass
    def get_classifier_type(self):
        """Return the type of classifier used for training/predicting"""
        return ClassifierTypeMLX.Transformer
