"""
KerasClassifierNary - n-ary classifier using Keras.

Behavior comes from utils.ClassifierKerasNary. This class adds membership
in the Predictors hierarchy (BaseClassifier -> BasePredictor) so strategy
code can target the Predictors namespace.
"""

from Predictors.KerasBaseClassifier import KerasBaseClassifier
from utils.ClassifierKerasNary import ClassifierKerasNary


class KerasClassifierNary(KerasBaseClassifier, ClassifierKerasNary):
    """N-ary Keras classifier. Drop-in replacement for utils.ClassifierKerasNary in the new hierarchy."""
    pass
