"""
MLXRegressorMultiHorizon — drop-in wrapper combining BaseRegressor with
utils.RegressorMLXMultiHorizon, mirroring MLXRegressorLinear's pattern.
"""

from Predictors.BaseRegressor import BaseRegressor
from utils.RegressorMLXMultiHorizon import RegressorMLXMultiHorizon


class MLXRegressorMultiHorizon(BaseRegressor, RegressorMLXMultiHorizon):
    """Multi-horizon MLX regressor wrapper. Drop-in replacement for
    utils.RegressorMLXMultiHorizon when used through the strategy's
    Predictor selection path."""
    pass
