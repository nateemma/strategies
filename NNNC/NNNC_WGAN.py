# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error
# flake8: noqa: F401, E402, F541, W0718, W0719

"""
NNNC_WGAN - Subclass of NNNCStrategy using WGAN-GP for high-fidelity augmentation.

Declares GANType.WGAN so the facade-aware base class can route augmentation
through GANFacade.  The heavy lifting (normalisation, one-hot encoding, etc.)
stays in BaseNNStrategy.wgan_enhance_training_data().
"""

from pandas import DataFrame
import numpy as np
from typing import Tuple

from NNNCStrategy import NNNCStrategy
from GANs.GANType import GANType


class NNNC_WGAN(NNNCStrategy):

    # Declare the GAN type so facade-aware callers can route correctly.
    gan_type = GANType.WGAN

    augment_training_data = True
    wgan_target_ratio = 0.8

    def enhance_training_data(
        self, train_df: DataFrame, train_labels: np.ndarray
    ) -> Tuple[DataFrame, np.ndarray]:
        """Augment 2-D training data with WGAN-GP (MLX-accelerated if available)."""
        return self.wgan_enhance_training_data(train_df, train_labels)
