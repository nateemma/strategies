# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""Random Forest subclass
"""

import sys
from pathlib import Path

# Make sibling Sklearn modules importable.
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from SklearnStrategy import SklearnStrategy  # noqa: E402
import SklearnClassifier  # noqa: E402


class Skl_RandomForest(SklearnStrategy):
    """
    Random Forest sklearn strategy.

    Sklearn classifiers work with 2D DataFrames (samples, features), not 3D
    tensors.  SklearnStrategy handles the DataFrame conversion; this subclass
    only selects the classifier type.
    """

    # Sklearn classifiers work with DataFrames directly, not tensors
    # seq_len is effectively 1 for sklearn (single timestep per sample)
    seq_len = 1

    def get_classifier_type(self):
        """Return the type of sklearn classifier used for training/predicting"""
        return SklearnClassifier.ClassifierType.RandomForest
