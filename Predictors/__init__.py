"""
Predictors package — task-type-organized predictor hierarchy.

This is a parallel hierarchy to utils/Classifier*.py. The existing utils/
classifiers remain authoritative for behavior; classes in this package
combine the new task-type bases (BasePredictor → BaseClassifier /
BaseRegressor / BaseAnomalyDetector) with the existing utils classifiers
via multiple inheritance.

Strategies should import from Predictors/* rather than utils/Classifier*.
"""
