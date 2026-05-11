# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""NNPredict_Ridge — sklearn Ridge regression baseline.

Linear in the (flattened) feature space. Outperformed the LSTM regressors
on Spearman rank correlation in early diagnostics, so kept as a working
predictor and a floor any nonlinear model should beat.
"""

import sys
from pathlib import Path
from typing import Any

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNPredictStrategy import NNPredictStrategy
import NNPredictRegressorRidge


class NNPredict_Ridge(NNPredictStrategy):
    def get_classifier_type(self):
        return NNPredictRegressorRidge.RegressorTypeRidge.RIDGE

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> Any:
        reg, _ = NNPredictRegressorRidge.create_regressor_ridge(
            classifier_type, pair, num_features, seq_len
        )
        return reg
