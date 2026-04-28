# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""XGBoost sklearn strategy with CTAB-GAN+ augmentation."""

import sys
from pathlib import Path

# Make sibling Sklearn modules importable.
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from SklearnStrategy import SklearnStrategy  # noqa: E402
import SklearnClassifier  # noqa: E402
from GANs.GANType import GANType  # noqa: E402


class Skl_XGBoost_CGP(SklearnStrategy):
    """XGBoost sklearn strategy with CTAB-GAN+ augmentation.

    Augmentation is dispatched through ``BaseNNStrategy.enhance_training_data``;
    this class only declares the GAN type and the augmentation ratio.
    """

    seq_len = 1  # Sklearn classifiers work on flat samples.

    # Single-task CTAB-GAN+ augmentation.
    gan_type = GANType.CTAB_GAN
    gan_target_ratio = 0.8

    # Use only real signals as the basis; GAN handles synthetic samples.
    augment_training_data = False

    def get_classifier_type(self):
        return SklearnClassifier.ClassifierType.XGBoost
