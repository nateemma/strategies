# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""XGBoost subclass with WGAN enhancement"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
from typing import Tuple

# Make sibling Sklearn modules importable.
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from SklearnStrategy import SklearnStrategy  # noqa: E402
import SklearnClassifier  # noqa: E402


class Skl_XGBoost_WGAN(SklearnStrategy):
    """
    XGBoost sklearn strategy with WGAN-GP augmentation.

    The training-time augmentation goes through
    BaseNNStrategy.wgan_enhance_training_data, which uses GANInterface
    under the hood — no direct WGAN module import is needed here.
    """

    # Sklearn classifiers work with DataFrames directly, not tensors
    # seq_len is effectively 1 for sklearn (single timestep per sample)
    seq_len = 1

    def get_classifier_type(self):
        """Return the type of sklearn classifier used for training/predicting"""
        return SklearnClassifier.ClassifierType.XGBoost

    # we only want 'real' signals, since we are augmenting anyway
    augment_training_data = False

    def enhance_training_data(
        self, train_df: DataFrame, train_labels: np.ndarray
    ) -> Tuple[DataFrame, np.ndarray]:
        """Optional hook to modify train/test tensors and labels before training.
        Typical use would be to run a GAN to generate more training data, or to apply a 
        custom data augmentation pipeline.

        Must return (train_data, train_labels).
        """

        return self.wgan_enhance_training_data(train_df, train_labels)

