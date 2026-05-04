"""
SklearnBaseClassifier - sklearn-backed base for classifier predictors.

Combines the BaseClassifier marker with utils.ClassifierSklearn. All
behavior comes from utils.ClassifierSklearn. The 18 strategy-side
SklearnClassifier_* concrete classes (LogisticRegression, DecisionTree,
RandomForest, GaussianNB, MLP, KNeighbors, GradientBoosting, AdaBoost,
LinearSVC, XGBoost, Voting, Stacking, etc.) inherit from this directly.
"""

from Predictors.BaseClassifier import BaseClassifier
from utils.ClassifierSklearn import ClassifierSklearn


class SklearnBaseClassifier(BaseClassifier, ClassifierSklearn):
    """sklearn-backed BaseClassifier. Use as the parent for sklearn Predictors classifier classes."""
    pass
