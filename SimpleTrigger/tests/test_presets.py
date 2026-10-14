"""T024 — presets are total and resolve to their reference frameworks (023 data-model, R1)."""

from __future__ import annotations

import importlib.util

import pytest
from evaluation import fingerprint as fp
from SimpleTrigger import presets as P
from SimpleTrigger.SimpleTriggerStrategy import SimpleTriggerStrategy


CFG = fp.load_config(fp.STRAT_DIR / "config" / "config_wf.json")


def _sha(cls, params_json=None):
    return fp.resolve(cls, params_json, CFG)[0]


def _arm_d_json_at_6_5h():
    """Arm D's .json with grace put back to 6.5 h. NNNC's default moved to 24 h on 2026-10-13
    (research/nnnc_framework/); ARM_D_OPEN keeps 6.5 h, so the twin now differs in that one
    value and must match everywhere else."""
    import copy

    j = copy.deepcopy(fp.load_params_json("NNNC_A_Raw_Open"))
    j["params"]["sell"]["stoploss_grace_hours"] = 6.5
    return j


def _arm_d_no_add():
    """Arm D with NNNC's in-profit held-pair add (on by default since 2026-10-13,
    research/held_pair/) put back to off. ARM_D_OPEN has no add."""
    ref = fp.load_strategy_class("NNNC_A_Raw_Open")
    return type("NNNC_A_Raw_OpenNoAdd", (ref,), {"HELD_PAIR_ADD": None})


def _triggeronly():
    spec = importlib.util.spec_from_file_location(
        "_to_ref", fp.STRAT_DIR / "SimpleStrategies" / "TriggerOnly.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.TriggerOnlyStrategy


def test_presets_are_total():
    for name in P.PRESETS:
        P.get(name)  # validates
    broken = P.get("ARM_D_OPEN")
    del broken["sell"]["cexit_take_profit"]
    with pytest.raises(P.PresetError):
        P.validate(broken)
    with pytest.raises(P.PresetError):
        P.get("NO_SUCH_PRESET")


def test_arm_d_open_resolves_to_arm_d():
    class _ArmD(SimpleTriggerStrategy):
        TRIGGER_NAME = "gbb"

    ref = _arm_d_no_add()
    assert _sha(_ArmD) == _sha(ref, _arm_d_json_at_6_5h())
    assert _sha(_ArmD) != _sha(ref, fp.load_params_json("NNNC_A_Raw_Open"))
    assert _sha(_ArmD) != _sha(fp.load_strategy_class("NNNC_A_Raw_Open"), _arm_d_json_at_6_5h())


def test_legacy_resolves_to_triggeronly():
    class _Legacy(SimpleTriggerStrategy):
        TRIGGER_NAME = "gbb"
        PRESET = "LEGACY_TRIGGERONLY"

    assert _sha(_Legacy) == _sha(_triggeronly())


def test_presets_differ_exactly_where_research_r1_says():
    a, b = P.get("ARM_D_OPEN")["attributes"], P.get("LEGACY_TRIGGERONLY")["attributes"]
    for k in (
        "use_atr_adaptive_stoploss",
        "atr_stoploss_multiplier",
        "atr_stoploss_floor",
        "atr_stoploss_cap",
        "startup_candle_count",
    ):
        assert a[k] != b[k], k
    assert a["use_atr_adaptive_stoploss"] is True and b["use_atr_adaptive_stoploss"] is False


def test_simple_trigger_gbb_is_arm_d_twin():
    # T033: the US3 twin shares arm D's framework (FR-013)
    gbb = fp.load_strategy_class("SimpleTrigger_Gbb")
    ref = _arm_d_no_add()
    assert _sha(gbb) == _sha(ref, _arm_d_json_at_6_5h())


@pytest.mark.parametrize(
    "legacy,original",
    [
        ("SimpleTrigger_Legacy_Breakout", "BreakoutOpen"),
        ("SimpleTrigger_Legacy_BreakoutVol", "BreakoutVolOpen"),
        ("SimpleTrigger_Legacy_BreakoutBb", "BreakoutBbOpen"),
        ("SimpleTrigger_Legacy_BreakoutGbb", "BreakoutGbbOpen"),
        ("SimpleTrigger_Legacy_Geometry", "GeometryOpen"),
        ("SimpleTrigger_Legacy_Gbb", "GbbOpen"),
        ("SimpleTrigger_Legacy_GbbRobust", "GbbRobustOpen"),
    ],
)
def test_legacy_configs_match_their_originals(legacy, original):
    # T028: same framework AND same trigger as the recorded 021/022 arm
    a, b = fp.load_strategy_class(legacy), fp.load_strategy_class(original)
    assert _sha(a) == _sha(b), legacy
    assert a.TRIGGER_NAME == b.TRIGGER_NAME


# --- feature 024 (T011): ARM_D_FAMILY = ARM_D_OPEN + per-family exit sets ---------------------


def test_arm_d_family_is_arm_d_open_plus_family_sets():
    fam, arm_d = P.get("ARM_D_FAMILY"), P.get("ARM_D_OPEN")
    for section in ("attributes", "buy", "sell"):
        assert fam[section] == arm_d[section], section
    assert fam["family_exits"] == {
        "reversion": {},
        "momentum": {"take_profit": 10.0, "unclog_12h_hours": 24.0},
        "unassigned": {},
    }


def test_shared_presets_resolve_every_family_to_no_overrides():
    for name in ("ARM_D_OPEN", "LEGACY_TRIGGERONLY"):
        assert P.family_exits(P.get(name)) == {
            "reversion": {}, "momentum": {}, "unassigned": {}
        }, name


def test_family_exits_with_an_unknown_family_is_refused():
    bad = P.get("ARM_D_FAMILY")
    bad["family_exits"]["sideways"] = {}
    with pytest.raises(P.PresetError, match="family"):
        P.validate(bad)
