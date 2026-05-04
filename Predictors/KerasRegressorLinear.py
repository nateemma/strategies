"""
KerasRegressorLinear - linear regressor using Keras.

Behavior comes from utils.ClassifierKerasLinear (which despite its name
is a regressor producing continuous outputs, not a classifier). This
class is the only regressor in the Predictors hierarchy today, so it
inherits BaseRegressor directly without an intervening framework base.
"""

from Predictors.BaseRegressor import BaseRegressor
from utils.ClassifierKerasLinear import ClassifierKerasLinear


class KerasRegressorLinear(BaseRegressor, ClassifierKerasLinear):
    """Linear Keras regressor. Drop-in replacement for utils.ClassifierKerasLinear."""
    pass
