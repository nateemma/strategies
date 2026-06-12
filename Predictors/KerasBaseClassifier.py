"""KerasBaseClassifier - Keras classifier base (MLX-style: infra + task marker).

Combines the task-agnostic Keras infrastructure (KerasBasePredictor) with the
BaseClassifier task marker. Concrete Keras classifiers inherit this.
"""

from Predictors.KerasBasePredictor import KerasBasePredictor
from Predictors.BaseClassifier import BaseClassifier


class KerasBaseClassifier(KerasBasePredictor, BaseClassifier):
    pass
