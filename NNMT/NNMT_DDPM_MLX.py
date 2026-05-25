# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNMT_DDPM_MLX - Subclass of NNMTStrategy using DDPM and MLX models
"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
from typing import Dict, Tuple
import mlx.core as mx

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMT_DDPM import NNMT_DDPM  # noqa: E402
from ClassifierKeras import ClassifierKeras
from NNMT.NNMTClassifierMLX import ClassifierTypeMLX, create_classifier_mlx

# -----------


class NNMT_DDPM_MLX(NNMT_DDPM):

    # Scalar broadcasts to every task — see balance_multi_task.target_ratios.
    gan_target_ratio = 0.5
    gan_run_diagnostics = True

    # default is LSTM type. Override get_classifier_type() in subclass
    def get_classifier_type(self):
        """Return the type of classifier used for training/predicting"""
        return ClassifierTypeMLX.LSTM

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
            self._apply_classifier_overrides(clf)
        else:
            print(
                "ERROR: This strategy requires Apple's MLX package, and only runs on native Apple hardware"
            )
            clf = None
        return clf
