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


@pytest.mark.parametrize("op,base", [
    ("NNPredict_Ridge_OP", "NNPredict_Ridge"), ("NNPredict_Ridge_SA_OP", "NNPredict_Ridge_SA"),
    ("NNPredict_LSTM_Raw_OP", "NNPredict_LSTM_Raw"), ("NNPredict_LSTM_RawInd_OP", "NNPredict_LSTM_RawInd"),
    ("NNPredict_LSTM_RawRet_OP", "NNPredict_LSTM_RawRet"), ("NNPredict_Ridge_Flip_OP", "NNPredict_Ridge")])
def test_operating_point_twin_reuses_its_base_weights(storage, op, base):
    import importlib

    o = getattr(importlib.import_module(op), op)
    b = getattr(importlib.import_module(base), base)
    assert MK.key_hash(MK.resolve(o, CONFIG)) == MK.key_hash(MK.resolve(b, CONFIG))
    assert o.magnitude_floor_mode == "percentile" and o.buy_params["entry_guard_threshold"] == 5.0


@pytest.mark.parametrize("arm", ["NNPredict_Ridge_Exc_OP", "NNPredict_Ridge_H24_OP"])
def test_threshold_amendment_retrains_get_their_own_weights(storage, arm):
    import importlib

    from NNPredict_Ridge import NNPredict_Ridge

    a = getattr(importlib.import_module(arm), arm)
    assert MK.key_hash(MK.resolve(a, CONFIG)) != MK.key_hash(MK.resolve(NNPredict_Ridge, CONFIG))
    assert a.magnitude_floor_mode == "percentile" and a.buy_params["entry_guard_threshold"] == 5.0
