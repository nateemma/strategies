"""
Predictors package — task-type-organized predictor hierarchy.

Classes are organized by task type (BasePredictor → BaseClassifier /
BaseRegressor / BaseAnomalyDetector) and hold the predictor implementations.
Strategies import their classifiers/regressors from here.
"""

# Self-bootstrap user_data/strategies AND utils/ onto sys.path so both the
# qualified ``from utils.X import ...`` imports and any remaining bare sibling
# imports resolve regardless of how the package is first loaded (e.g. pytest
# collection). Done explicitly here rather than relying on a module-load
# side-effect of any one classifier file.
import sys as _sys
from pathlib import Path as _Path
_STRAT_DIR = _Path(__file__).resolve().parent.parent
for _p in (str(_STRAT_DIR), str(_STRAT_DIR / "utils")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
