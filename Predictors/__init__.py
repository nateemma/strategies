"""
Predictors package — task-type-organized predictor hierarchy.

This is a parallel hierarchy to utils/Classifier*.py. The existing utils/
classifiers remain authoritative for behavior; classes in this package
combine the new task-type bases (BasePredictor → BaseClassifier /
BaseRegressor / BaseAnomalyDetector) with the existing utils classifiers
via multiple inheritance.

Strategies should import from Predictors/* rather than utils/Classifier*.
"""

# Side-effect import: utils/ClassifierKeras.py runs sys.path.append(utils/)
# at module load. Several utils/Classifier*.py files use sibling-style
# imports (e.g. `from ClassifierKeras import ClassifierKeras`) that need
# utils/ on sys.path to resolve. Loading ClassifierKeras here once when
# the Predictors package is first imported guarantees the path is set up
# before any Predictors module reaches `from utils.ClassifierX import X`.
# Do NOT remove (and do not let an automated lint cleanup remove it).
from utils.ClassifierKeras import ClassifierKeras  # noqa: F401
