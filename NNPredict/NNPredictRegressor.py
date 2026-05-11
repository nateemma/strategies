# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
Neural Network Regressor factory for NNPredict strategies.

Mirrors NNNClassifier.create_classifier()/ClassifierType so the strategy
selection pattern is consistent across the classifier and regressor families.
The actual model code lives in utils/NNPredictors.py — this module just
exposes a parallel enum + factory.

Usage:
    reg, name = NNPredictRegressor.create_regressor(
        NNPredictRegressor.RegressorType.LSTM, pair, nfeatures, seq_len
    )
"""

import sys
from pathlib import Path
from enum import Enum

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from utils.NNPredictors import predictor_lstm


class RegressorType(Enum):
    LSTM = predictor_lstm  # Long-Short Term Memory (basic regressor)


def create_regressor(reg_type: RegressorType, pair, nfeatures, seq_len, tag=""):
    """Factory — returns (regressor, name)."""
    name = str(reg_type).split(".")[-1]
    # NNPredictor signature is (pair, seq_len, num_features) — different arg
    # order from NNNClassifier's (pair, seq_len, nfeatures, tag=...).
    reg = reg_type.value(pair, seq_len, nfeatures)
    return reg, name
