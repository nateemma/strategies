"""
Predictors package — task-type-organized predictor hierarchy.

This is a parallel hierarchy to utils/Classifier*.py. The existing utils/
classifiers remain authoritative for behavior; classes in this package
combine the new task-type bases (BasePredictor → BaseClassifier /
BaseRegressor / BaseAnomalyDetector) with the existing utils classifiers
via multiple inheritance.

Strategies should import from Predictors/* rather than utils/Classifier*.
"""

# Self-bootstrap user_data/strategies on sys.path so the side-effect
# import below works in contexts (e.g. pytest collection) that don't
# already have it set.
import sys as _sys
from pathlib import Path as _Path
_STRAT_DIR = str(_Path(__file__).resolve().parent.parent)
if _STRAT_DIR not in _sys.path:
    _sys.path.insert(0, _STRAT_DIR)

# Side-effect import: utils/ClassifierKeras.py runs sys.path.append(utils/)
# at module load. Several utils/Classifier*.py files use sibling-style
# imports (e.g. `from ClassifierKeras import ClassifierKeras`) that need
# utils/ on sys.path to resolve. Loading ClassifierKeras here once when
# the Predictors package is first imported guarantees the path is set up
# before any Predictors module reaches `from utils.ClassifierX import X`.
# Do NOT remove (and do not let an automated lint cleanup remove it).
from utils.ClassifierKeras import ClassifierKeras  # noqa: F401
