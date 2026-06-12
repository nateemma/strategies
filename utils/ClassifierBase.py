# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# type: ignore
# pylint: disable=import-error

"""
ClassifierBase - shared base for the classifier backends.

ClassifierKeras / ClassifierMLX / ClassifierSklearn / ClassifierDarts /
ClassifierPyTorch were independent classes that each re-declared the same small
capability stubs. Those byte-identical methods are hoisted here so the backends
inherit one copy. Methods that genuinely differ per backend stay overridden on
the backend (and shadow these defaults via normal MRO).

This base is intentionally state-light: the hoisted methods read instance
attributes (clean_data_required, requires_dataframes, prescale_dataframe,
single_prediction, is_trained) that each backend's __init__ already sets.
"""

import os
from pathlib import Path

import numpy as np


class ClassifierBase:

    # --- capability flags (identical across all five backends) ---

    def needs_clean_data(self) -> bool:
        # print("    clean_data_required: ", self.clean_data_required)
        return self.clean_data_required

    def needs_dataframes(self) -> bool:
        return self.requires_dataframes

    def prescale_data(self) -> bool:
        return self.prescale_dataframe

    def returns_single_prediction(self) -> bool:
        return self.single_prediction

    # --- training state (Sklearn/Darts/PyTorch default; Keras/MLX override) ---

    def model_is_trained(self) -> bool:
        return self.is_trained

    # --- model storage location (Sklearn/Darts/PyTorch default; Keras/MLX
    # override). __file__ lives in utils/, the same dir the original definers
    # lived in, so the resolved path (utils/models/) is unchanged. ---

    def get_model_root_dir(self):
        # set as subdirectory of location of this file (so that it can be included in the repository)
        file_dir = os.path.dirname(str(Path(__file__)))
        root_dir = file_dir + "/models/"
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        return root_dir

    # --- misc shared utility ---

    def mad_score(self, points):
        """https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm """
        m = np.median(points)
        ad = np.abs(points - m)
        mad = np.median(ad)

        return 0.6745 * ad / mad
