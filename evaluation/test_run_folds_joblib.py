"""039 T005 (research R2): a Ridge fold saves `.joblib` (Predictors/RidgeRegressor.py), and the fold
runner's training gate (`run_folds.fold_has_model` -> `ModelKey.weights_exist`) must accept it, or every
Ridge fold would be reported as having NO MODEL ARTEFACT."""

from __future__ import annotations

import sys
from pathlib import Path

STRAT = Path(__file__).resolve().parents[1]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from Framework import ModelKey as MK  # noqa: E402


def test_joblib_counts_as_trained_weights(tmp_path):
    stem = tmp_path / "model"
    assert not MK.weights_exist(stem)
    Path(f"{stem}.joblib").write_bytes(b"ridge")
    assert MK.weights_exist(stem)


def test_ridge_regressor_writes_a_suffix_the_gate_accepts():
    src = (STRAT / "Predictors" / "RidgeRegressor.py").read_text()
    assert '".joblib"' in src and ".joblib" in MK.WEIGHT_SUFFIXES
