"""
BaseRegressor - base for predictors that produce continuous values.

Marker class. Concrete regressors combine this with a framework-specific
implementation.
"""

from Predictors.BasePredictor import BasePredictor


class BaseRegressor(BasePredictor):
    """Marker for regressor-style predictors. Carries no behavior on its own."""
    pass
