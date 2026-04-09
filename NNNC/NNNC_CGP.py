# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNNC_CGP - Subclass of NNNCStrategy using CTAB-GAN+ for high-fidelity augmentation
"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
from typing import Tuple

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNCStrategy import NNNCStrategy  # noqa: E402

# -----------


class NNNC_CGP(NNNCStrategy):

    augment_training_data = True

    # CTAB-GAN+ configuration
    # amount of augmentation. Don't set above 1.0 or the model will over-fit
    cgp_augmentation_target_ratio = 0.8

    # -----------

    def enhance_training_data(
        self, train_df: DataFrame, train_labels: np.ndarray
    ) -> Tuple[DataFrame, np.ndarray]:
        """Optional hook to modify train/test tensors and labels before training.
        Uses CTAB-GAN+ to generate more training data

        Must return (train_data, train_labels).
        """
        return self.ctab_gan_enhance_training_data(train_df, train_labels)
