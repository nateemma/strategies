# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
NNNC_WGAN - Subclass of NNNCStrategy using WGAN-GP for high-fidelity augmentation
"""

from pandas import DataFrame
import numpy as np
from typing import Tuple

from NNNCStrategy import NNNCStrategy


class NNNC_WGAN(NNNCStrategy):

    augment_training_data = True
    wgan_target_ratio = 0.8

    def enhance_training_data(
        self, train_df: DataFrame, train_labels: np.ndarray
    ) -> Tuple[DataFrame, np.ndarray]:
        """Optional hook to modify train/test tensors and labels before training.
        Uses WGAN-GP (MLX-accelerated if available) to generate more training data.
        """
        return self.wgan_enhance_training_data(train_df, train_labels)

    # def preprocess_training_data(
    #     self, dataframe: DataFrame, train_data, test_data, train_labels, test_labels
    # ):
    #     return self.wgan_preprocess_training_data(
    #         dataframe, train_data, test_data, train_labels, test_labels
    #     )
