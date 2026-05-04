"""
KerasAnomalyDetector - anomaly detector using Keras.

Behavior comes from utils.ClassifierKerasAnomaly. This is the only
anomaly-detection class in the Predictors hierarchy today, so it
inherits BaseAnomalyDetector directly without an intervening framework
base.
"""

import sys
import os
# Add utils to path to support ClassifierKerasAnomaly's relative imports
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from Predictors.BaseAnomalyDetector import BaseAnomalyDetector
from utils.ClassifierKerasAnomaly import ClassifierKerasAnomaly


class KerasAnomalyDetector(BaseAnomalyDetector, ClassifierKerasAnomaly):
    """Keras anomaly detector. Drop-in replacement for utils.ClassifierKerasAnomaly."""
    pass
