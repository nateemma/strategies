"""039 T003 (research R1): NNPredict's post-model gate settings must NOT move the model key (they act
after `classifier.predict`); the label settings must."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "NNPredict"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from Framework import ModelKey as MK  # noqa: E402
from Framework.BaseNNStrategy import BaseNNStrategy  # noqa: E402
from NNPredict.NNPredictStrategy import NNPredictStrategy  # noqa: E402

CONFIG = {"exchange": {"pair_whitelist": ["SOL/USDT", "LINK/USDT"]}, "timeframe": "15m"}

POST_MODEL = {"entry_z": 2.0, "min_magnitude": 0.3, "rolling_window": 100, "magnitude_floor_mode": "percentile",
              "magnitude_pctile": 0.8, "mag_std_mult": 1.5, "entry_quantile": 0.95, "exit_quantile": 0.05}
WEIGHT_SHAPING = {"target_mode": "excursion", "target_max_gain": 6.0, "atr_floor": 2e-3, "LABEL_SMOOTH_WINDOW": 2}


@pytest.fixture
def storage(tmp_path, monkeypatch):
    (tmp_path / "main_scaler.pkl").write_bytes(b"scaler-v1")
    monkeypatch.setattr(BaseNNStrategy, "get_storage_location", lambda self: str(tmp_path) + "/")
    monkeypatch.delenv("FT_MODEL_EVAL_STRICT", raising=False)
    monkeypatch.delenv("MLX_SEED", raising=False)
    return tmp_path


def _h(**attrs):
    return MK.key_hash(MK.resolve(type("Stub", (NNPredictStrategy,), dict(attrs)), CONFIG))


@pytest.mark.parametrize("name,value", list(POST_MODEL.items()))
def test_post_model_gate_settings_leave_the_key(storage, name, value):
    assert _h(**{name: value}) == _h()


@pytest.mark.parametrize("name,value", list(WEIGHT_SHAPING.items()))
def test_label_settings_move_the_key(storage, name, value):
    assert _h(**{name: value}) != _h()
