"""
MLXClassifierMultiTask - multi-task classifier using MLX (Apple Silicon).

Behavior comes from utils.ClassifierMLXMultiTask.
"""

from Predictors.MLXBaseClassifier import MLXBaseClassifier
from utils.ClassifierMLXMultiTask import ClassifierMLXMultiTask


class MLXClassifierMultiTask(MLXBaseClassifier, ClassifierMLXMultiTask):
    """Multi-task MLX classifier. Drop-in replacement for utils.ClassifierMLXMultiTask."""
    pass
