# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""NNPredict_LSTM — basic LSTM regressor for the NNPredict family."""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNPredictStrategy import NNPredictStrategy
import NNPredictRegressor


class NNPredict_LSTM(NNPredictStrategy):

    def get_classifier_type(self):
        return NNPredictRegressor.RegressorType.LSTM
