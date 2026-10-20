"""040 T013 (research R2): NNMT's model key holds only what shapes the weights. Post-model settings
(thresholds, filters applied to the predictions, exit settings, framework switches) leave the key
unchanged, so a variant that changes them reuses the trained weights; anything weight-shaping changes
it. An unpinned NNMT class is a mainline model (any pairlist finds it)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "NNMT"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from Framework import ModelKey as MK  # noqa: E402
from NNMT.NNMT_MLX import NNMT_MLX  # noqa: E402

CONFIG = {"exchange": {"pair_whitelist": ["SOL/USDT", "LINK/USDT"]}, "timeframe": "15m"}


def _key(**attrs):
    cls = type("NNMT_KeyProbe", (NNMT_MLX,), {"MODEL_TRAIN_TIMERANGE": "20210101-20230101", **attrs})
    return MK.key_hash(MK.resolve(cls, CONFIG))


@pytest.fixture(scope="module")
def base():
    return _key()


@pytest.mark.parametrize("name, value", [
    ("prediction_threshold", "probe"),
    ("profit_prediction_threshold", "probe"),
    ("min_consecutive_buys", "probe"),
    ("apply_task_filters", "probe"),
    ("forward_peak_regressor_threshold", "probe"),
    ("use_forward_peak_regressor_filter", True),
    ("enable_exit_signal", "probe"),
    ("exit_rvol_threshold", "probe"),
    ("bailout_min_loss", -0.5),
    ("ENTRY_MODE", "trigger_and_model"),
    ("TRIGGER_NAME", "gbb"),
    ("HELD_PAIR_ADD", {"in_profit_only": True}),
    ("take_profit_atr_mult", 2.897),
])
def test_post_model_settings_share_the_weights(base, name, value):
    assert _key(**{name: value}) == base


@pytest.mark.parametrize("name, value", [
    ("_CLASSIFIER_TASK_WEIGHTS", {"trading": 1.0, "regime": 0.0, "risk": 0.0, "momentum": 0.0,
                                  "flow": 0.0, "profit": 0.0}),
    ("PROFIT_HORIZON", 16),
    ("regime_lookback", 40),
])
def test_weight_shaping_settings_change_the_key(base, name, value):
    assert _key(**{name: value}) != base


def test_classifier_type_changes_the_key(base):
    from NNMT.NNMTClassifierMLX import ClassifierTypeMLX

    other = next(t for t in ClassifierTypeMLX if t != NNMT_MLX.classifier_type)
    assert _key(classifier_type=other) != base


def test_an_unpinned_nnmt_class_is_mainline():
    cls = type("NNMT_MainProbe", (NNMT_MLX,), {})
    assert MK.is_mainline(cls)
    loc = MK.locate_for(cls, {"exchange": {"pair_whitelist": []}, "timeframe": "15m"}, "NNMT_MainProbe")
    assert loc.kind in ("mainline-absent", "mainline", "legacy", "keyed", "migrated")
    assert loc.key["run"]["whitelist"] == MK.DYNAMIC_WHITELIST
