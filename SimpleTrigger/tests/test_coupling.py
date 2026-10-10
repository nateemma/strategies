"""T049 — coupled exit mode is a pure parameter mapping (023 research R8, FR-015/FR-016)."""

from __future__ import annotations

import pytest
from Framework.TrainingSignals import LabelMethod
from SimpleTrigger import coupling
from SimpleTrigger.presets import PresetError
from utils.ExitPolicy import net_from_price_ratio, price_ratio_for_net


def _cls(method, **targets):
    return type(
        "X",
        (),
        {
            "LABEL_METHOD": method,
            "LABEL_TARGETS": targets,
            "FEE_PER_SIDE": 0.001,
            "timeframe": "15m",
        },
    )


def test_gbb_maps_take_profit_and_max_hold_but_has_no_loss_barrier():
    g, h, f = 0.02, 960, 0.001
    res = coupling.resolve(_cls(LabelMethod.gbb, min_gain=g, horizon=h))
    assert res["sell"]["cexit_take_profit"] == round(net_from_price_ratio(1 + g, f), 12)
    assert res["sell"]["cexit_max_days"] == h * 15 // 1440 == 10
    assert res["attributes"] == {}
    assert res["record"]["coupled_stop"] == "none-in-label"


def test_forward_mae_maps_its_drawdown_to_a_static_stop():
    res = coupling.resolve(_cls(LabelMethod.forward_mae, min_gain=0.015, horizon=96, loss=0.10))
    assert res["attributes"] == {"stoploss": -0.10, "use_atr_adaptive_stoploss": False}
    assert res["record"]["coupled_stop"] == "label"


def test_loss_target_ignored_for_a_label_without_a_loss_barrier():
    with pytest.raises(PresetError, match="no loss barrier"):
        coupling.resolve(_cls(LabelMethod.gbb, min_gain=0.02, horizon=960, loss=0.05))


def test_unsupported_label_raises():
    with pytest.raises(PresetError):
        coupling.resolve(_cls(LabelMethod.local_min_followthrough, min_gain=0.02, horizon=960))


def test_horizon_must_be_whole_days():
    with pytest.raises(PresetError, match="whole days"):
        coupling.resolve(_cls(LabelMethod.gbb, min_gain=0.02, horizon=12))


def test_round_trip_reproduces_the_decoupled_take_profit():
    tp, f = 0.022, 0.001
    g = price_ratio_for_net(tp, f) - 1.0
    res = coupling.resolve(_cls(LabelMethod.gbb, min_gain=g, horizon=960))
    assert res["sell"]["cexit_take_profit"] == tp  # exact, not approximate


def test_missing_targets_raise():
    with pytest.raises(PresetError):
        coupling.resolve(_cls(LabelMethod.gbb, min_gain=0.02))
    with pytest.raises(PresetError):
        coupling.resolve(type("X", (), {"LABEL_METHOD": None, "LABEL_TARGETS": {}}))


def test_gbb_coupled_config_has_the_decoupled_framework():
    # T051 / SC-007 precondition
    from evaluation import fingerprint as fp

    cfg = fp.load_config(fp.STRAT_DIR / "config" / "config_wf.json")
    c = fp.load_strategy_class("SimpleTrigger_GbbCoupled")
    d = fp.load_strategy_class("SimpleTrigger_Gbb")
    assert fp.resolve(c, None, cfg)[0] == fp.resolve(d, None, cfg)[0]
    assert c.COUPLED_RESOLUTION["coupled_stop"] == "none-in-label"
