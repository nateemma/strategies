# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNMT_CGP_MLX_MultiLSTM - Subclass of NNMTStrategy using Ctab Gan and MLX models
"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
from typing import Dict, Tuple
import mlx.core as mx

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMT_CGP import NNMT_CGP  # noqa: E402
from ClassifierKeras import ClassifierKeras
from NNMT.NNMTClassifierMLX import ClassifierTypeMLX, create_classifier_mlx

# -----------


class NNMT_CGP_MLX_MultiLSTM(NNMT_CGP):

    # default is LSTM type. Override get_classifier_type() in subclass
    def get_classifier_type(self):
        """Return the type of classifier used for training/predicting"""
        return ClassifierTypeMLX.Multi_LSTM

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> ClassifierKeras:
        """Return the classifier used for training/predicting.

        The multi-task MLX factory does not take an ``nclasses`` argument —
        each of the six task heads emits a fixed 3-way softmax — so the call
        site here matches the Keras MT factory (NNMTClassifier.create_classifier)
        rather than the single-task NNNClassifierMLX one.
        """
        if hasattr(mx, "metal") and mx.metal.is_available():
            clf, _ = create_classifier_mlx(
                classifier_type, pair, num_features, seq_len
            )
        else:
            print(
                "ERROR: This strategy requires Apple's MLX package, and only runs on native Apple hardware"
            )
            clf = None
        return clf
