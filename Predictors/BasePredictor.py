"""
BasePredictor - root of the Predictors hierarchy.

Subclasses are organized by task type: classifiers (discrete labels),
regressors (continuous values), and anomaly detectors (one-class /
reconstruction-based scoring).

This root carries the small capability stubs shared by every predictor backend
(previously in utils/ClassifierBase, the B3 extraction). Backend bases override
the ones that genuinely differ (e.g. Keras/MLX override model_is_trained and
get_model_root_dir).
"""

import os
from pathlib import Path

import numpy as np


class BasePredictor:

    # --- capability flags (identical across all backends) ---

    def needs_clean_data(self) -> bool:
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
    # override). Anchored explicitly to <strategies>/utils/models to preserve
    # the path from when this lived in utils/ClassifierBase — moving the class
    # to Predictors/ must NOT change where models are written. ---

    def get_model_root_dir(self):
        strat_dir = os.path.dirname(os.path.dirname(str(Path(__file__))))
        root_dir = os.path.join(strat_dir, "utils", "models") + "/"
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
