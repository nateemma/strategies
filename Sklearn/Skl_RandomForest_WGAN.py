# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""Random Forest sklearn strategy with WGAN-GP augmentation."""

import sys
from pathlib import Path

# Make sibling Sklearn modules importable.
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from SklearnStrategy import SklearnStrategy  # noqa: E402
import SklearnClassifier  # noqa: E402
from GANs.GANType import GANType  # noqa: E402


class Skl_RandomForest_WGAN(SklearnStrategy):
    """Random Forest sklearn strategy with WGAN-GP augmentation.

    Augmentation is dispatched through ``BaseNNStrategy.enhance_training_data``;
    this class only declares the GAN type and the augmentation ratio.
    """

    # Sklearn classifiers work with DataFrames directly, not tensors;
    # seq_len is effectively 1 for sklearn (single timestep per sample).
    seq_len = 1

    # Single-task WGAN-GP augmentation.
    gan_type = GANType.WGAN
    gan_target_ratio = 0.8

    # Use only real signals as the basis; GAN provides synthetic samples
    # so layered signal augmentation would double-count.
    augment_training_data = False

    def get_classifier_type(self):
        return SklearnClassifier.ClassifierType.RandomForest
