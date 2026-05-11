"""
MLXRegressorLinear - linear regressor using MLX.

Drop-in wrapper combining BaseRegressor with utils.RegressorMLXLinear, in
the same shape as Predictors.KerasRegressorLinear (which combines
BaseRegressor with utils.ClassifierKerasLinear).
"""

from Predictors.BaseRegressor import BaseRegressor
from utils.RegressorMLXLinear import RegressorMLXLinear


class MLXRegressorLinear(BaseRegressor, RegressorMLXLinear):
    """Linear MLX regressor. Drop-in replacement for utils.RegressorMLXLinear."""
    pass
