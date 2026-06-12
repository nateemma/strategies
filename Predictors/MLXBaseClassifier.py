"""
MLXBaseClassifier - MLX classifier base.

Combines the task-agnostic MLX infrastructure (MLXBasePredictor) with the
BaseClassifier task marker. Concrete MLX classifiers (MLXClassifierNary,
MLXClassifierMultiTask) inherit this.
"""

from Predictors.MLXBasePredictor import MLXBasePredictor
from Predictors.BaseClassifier import BaseClassifier


class MLXBaseClassifier(MLXBasePredictor, BaseClassifier):
    pass
