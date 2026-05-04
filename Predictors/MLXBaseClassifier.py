"""
MLXBaseClassifier - MLX-backed base for classifier predictors.

Combines the BaseClassifier marker with utils.ClassifierMLX. All
behavior comes from utils.ClassifierMLX.
"""

from Predictors.BaseClassifier import BaseClassifier
from utils.ClassifierMLX import ClassifierMLX


class MLXBaseClassifier(BaseClassifier, ClassifierMLX):
    """MLX-backed BaseClassifier. Use as the parent for MLX Predictors classifier classes."""
    pass
