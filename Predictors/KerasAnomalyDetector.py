"""
KerasAnomalyDetector - anomaly detector using Keras.

Behavior comes from utils.ClassifierKerasAnomaly. This is the only
anomaly-detection class in the Predictors hierarchy today, so it
inherits BaseAnomalyDetector directly without an intervening framework
base.
"""

# Import ClassifierKeras first so its sys.path.append() runs and adds utils/
# to the path. ClassifierKerasAnomaly relies on sibling-style imports
# (e.g. `from ClassifierKeras import ClassifierKeras`) that need utils/ on
# sys.path.
from utils.ClassifierKeras import ClassifierKeras  # noqa: F401

from Predictors.BaseAnomalyDetector import BaseAnomalyDetector
from utils.ClassifierKerasAnomaly import ClassifierKerasAnomaly


class KerasAnomalyDetector(BaseAnomalyDetector, ClassifierKerasAnomaly):
    """Keras anomaly detector. Drop-in replacement for utils.ClassifierKerasAnomaly."""
    pass
