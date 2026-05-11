from __future__ import annotations

# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
Ridge regressor factory for NNPredict strategies.

Mirrors NNPredictRegressorMLX.create_regressor_mlx() and the Keras
NNPredictRegressor.create_regressor() — same shape so the strategy can swap
backends via get_classifier_type()/get_classifier() without touching the
strategy machinery.
"""

import sys
from pathlib import Path
from enum import Enum

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from Predictors.RidgeRegressor import RidgeRegressor


class RegressorTypeRidge(Enum):
    RIDGE = RidgeRegressor


def create_regressor_ridge(
    reg_type: RegressorTypeRidge,
    pair: str,
    nfeatures: int,
    seq_len: int,
    tag: str = "",
):
    name = str(reg_type).split(".")[-1]
    reg = reg_type.value(pair, seq_len, nfeatures, tag=tag)
    return reg, name
