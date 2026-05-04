"""
BaseClassifier - base for predictors that produce discrete class labels.

Marker class. Concrete classifiers (e.g. KerasClassifierNary,
SklearnBaseClassifier) inherit from this plus a framework-specific
implementation from utils/Classifier*.
"""

from Predictors.BasePredictor import BasePredictor


class BaseClassifier(BasePredictor):
    """Marker for classifier-style predictors. Carries no behavior on its own."""
    pass
