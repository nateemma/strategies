"""
KerasClassifierMultiTask - multi-task classifier using Keras.

Behavior comes from utils.ClassifierKerasMultiTask. Used by the NNMT
family for the six-task multi-head Trading/Regime/Risk/Flow/Momentum/
Profit predictor.
"""

from Predictors.KerasBaseClassifier import KerasBaseClassifier
from utils.ClassifierKerasMultiTask import ClassifierKerasMultiTask


class KerasClassifierMultiTask(KerasBaseClassifier, ClassifierKerasMultiTask):
    """Multi-task Keras classifier. Drop-in replacement for utils.ClassifierKerasMultiTask."""
    pass
