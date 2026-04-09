# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNMT_CGP - Subclass of NNMTStrategy using Multi-Task CTAB-GAN+ for high-fidelity augmentation
"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
from typing import Dict, Tuple

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMTStrategy import NNMTStrategy  # noqa: E402

# -----------


class NNMT_CGP(NNMTStrategy):



    augment_training_data = True 

    # CTAB-GAN+ configuration
    # cgp_augmentation_target_ratio = 0.4  # Augment minority classes to % of majority class size
    cgp_augmentation_target_ratio = 0.8

    batch_size = 2048  # bigger since we enhanced the data

    # -----------

    # -----------

    def enhance_training_data(
        self, train_df: DataFrame, train_labels: Dict[str, np.ndarray]
    ) -> Tuple[DataFrame, Dict[str, np.ndarray]]:
        """Optional hook to modify train/test tensors and labels before training.
        Uses Multi-Task CTAB-GAN+ to generate more training data

        Must return (train_data, train_labels).
        """
        return self.mt_ctab_gan_enhance_training_data(train_df, train_labels)
