"""
MLXClassifierNary - n-ary classifier using MLX (Apple Silicon).

Behavior comes from utils.ClassifierMLXNary.
"""

from Predictors.MLXBaseClassifier import MLXBaseClassifier
from utils.ClassifierMLXNary import ClassifierMLXNary


class MLXClassifierNary(MLXBaseClassifier, ClassifierMLXNary):
    """N-ary MLX classifier. Drop-in replacement for utils.ClassifierMLXNary."""
    pass
