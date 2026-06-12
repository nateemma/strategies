"""
BaseAnomalyDetector - base for predictors that score anomalies.

Marker class. Anomaly detection has a different training paradigm
(one-class, reconstruction-based) than ordinary classification, so it
sits as a sibling of BaseClassifier under BasePredictor rather than
underneath it. Concrete detectors combine this with a framework-specific
implementation.
"""

from Predictors.BasePredictor import BasePredictor


class BaseAnomalyDetector(BasePredictor):
    """Marker for anomaly-detection predictors. Carries no behavior on its own."""
    pass
