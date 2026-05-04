"""
BasePredictor - root of the Predictors hierarchy.

Marker class. Subclasses are organized by task type:
  - BaseClassifier (discrete-label prediction)
  - BaseRegressor (continuous-value prediction)
  - BaseAnomalyDetector (one-class / reconstruction-based scoring)

Concrete predictor classes combine a task-type base with a framework-
specific implementation (e.g. utils.ClassifierKeras) via multiple
inheritance. See Predictors/__init__.py for the design rationale.
"""


class BasePredictor:
    """Root marker class for all predictors. Carries no behavior on its own."""
    pass
