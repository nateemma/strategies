"""
BaseClassifier - base for predictors that produce discrete class labels.

Marker class. Concrete classifiers inherit from this plus a
framework-specific implementation.
"""

from Predictors.BasePredictor import BasePredictor


class BaseClassifier(BasePredictor):
    """Marker for classifier-style predictors. Carries no behavior on its own."""
    pass
