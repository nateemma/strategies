"""
BaseRegressor - base for predictors that produce continuous values.

Marker class. The current single concrete is KerasRegressorLinear
(combining BaseRegressor with utils.ClassifierKerasLinear).
"""

from Predictors.BasePredictor import BasePredictor


class BaseRegressor(BasePredictor):
    """Marker for regressor-style predictors. Carries no behavior on its own."""
    pass
