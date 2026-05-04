"""
KerasBaseClassifier - Keras-backed base for classifier predictors.

Combines the BaseClassifier marker with utils.ClassifierKeras. All
behavior comes from utils.ClassifierKeras; this class adds membership
in the Predictors hierarchy so isinstance checks against BaseClassifier
and BasePredictor succeed.
"""

from Predictors.BaseClassifier import BaseClassifier
from utils.ClassifierKeras import ClassifierKeras


class KerasBaseClassifier(BaseClassifier, ClassifierKeras):
    """Keras-backed BaseClassifier. Use as the parent for Keras Predictors classifier classes."""
    pass
