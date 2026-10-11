"""T003 — the framework fingerprint (023 research R4, contract C3/C5).

`exit_config_hash` hashes only a strategy's params JSON, so every class-attribute arm (all of
021's) hashed identically whatever its real exit was. The fingerprint hashes the RESOLVED
framework instead: class attributes across the MRO, hyperopt parameter defaults, the JSON merged
the way freqtrade merges it, and the run config. These tests pin each of those layers.
"""

from __future__ import annotations

from evaluation import fingerprint as fp


CONFIG = fp.STRAT_DIR / "config" / "config_wf.json"


def _resolve(name, params_json="auto"):
    cls = fp.load_strategy_class(name)
    pj = fp.load_params_json(name) if params_json == "auto" else params_json
    return fp.resolve(cls, pj, fp.load_config(CONFIG))


def test_fold_class_shares_arm_d_framework():
    # (a) a per-fold class differs from its base only in prediction_threshold, which is excluded
    sha_base, _ = _resolve("NNNC_A_Raw_Open")
    sha_fold, _ = _resolve("NNNC_ARO_f00")
    assert sha_base == sha_fold


def test_resolved_dict_covers_every_preset_member():
    # (b)
    _, resolved = _resolve("NNNC_A_Raw_Open")
    for key in fp.ATTRIBUTE_MEMBERS:
        assert key in resolved["attributes"], key
    for key in (
        "entry_adx_threshold",
        "entry_atr_pct",
        "entry_bb_width_threshold",
        "entry_close_norm_threshold",
        "entry_guard_threshold",
        "entry_rvol_threshold",
        "entry_enable_guards",
    ):
        assert key in resolved["buy"], key
    for key in (
        "cexit_take_profit",
        "cexit_max_days",
        "cexit_enable_profit_checks",
        "enable_exit_signal",
        "stoploss_grace_hours",
        "stoploss_grace_level",
    ):
        assert key in resolved["sell"], key
    for key in ("max_open_trades", "stake_amount", "dry_run_wallet", "fee"):
        assert key in resolved["run"], key


def test_entry_signal_settings_are_excluded():
    # (c) what a twin is ALLOWED to differ in never enters the hash
    _, resolved = _resolve("NNNC_A_Raw_Open")
    flat = repr(resolved)
    for excluded in ("prediction_threshold", "classifier_type", "LABEL_PARAMS", "MIN_BUY"):
        assert excluded not in flat, excluded


def test_one_stop_attribute_changes_the_sha():
    # (d)
    base = fp.load_strategy_class("NNNC_A_Raw_Open")
    pj = fp.load_params_json("NNNC_A_Raw_Open")
    cfg = fp.load_config(CONFIG)
    sha0, _ = fp.resolve(base, pj, cfg)
    tweaked = type("Tweaked", (base,), {"atr_stoploss_floor": -0.07})
    sha1, _ = fp.resolve(tweaked, pj, cfg)
    assert sha0 != sha1


def test_json_sell_override_is_merged_like_freqtrade():
    # (e) file values win over the class dict and the parameter default; others are kept
    base = fp.load_strategy_class("NNNC_A_Raw_Open")
    cfg = fp.load_config(CONFIG)
    pj = {"params": {"sell": {"cexit_take_profit": 0.031}}}
    _, resolved = fp.resolve(base, pj, cfg)
    assert resolved["sell"]["cexit_take_profit"] == 0.031
    _, no_json = fp.resolve(base, None, cfg)
    assert resolved["sell"]["cexit_max_days"] == no_json["sell"]["cexit_max_days"]
    # roi / stoploss / trailing / max_open_trades sections replace the attribute (hyper.py:60-84)
    pj2 = {"params": {"stoploss": {"stoploss": -0.09}, "max_open_trades": {"max_open_trades": 4}}}
    _, r2 = fp.resolve(base, pj2, cfg)
    assert r2["attributes"]["stoploss"] == -0.09
    assert r2["run"]["max_open_trades"] == 4


def test_resolve_never_instantiates(monkeypatch):
    # (f) instantiating an NNNC class loads, or TRAINS, a model
    from Framework import BaseNNStrategy as bnn

    def boom(*a, **k):
        raise AssertionError("resolve() instantiated the strategy")

    monkeypatch.setattr(bnn.BaseNNStrategy, "__init__", boom)
    _resolve("NNNC_A_Raw_Open")


def test_hold_window_changes_the_sha_and_unset_leaves_shas_alone():
    # C9: the hold-window mixin changes exits, so it must change the sha; strategies without it
    # must keep their recorded sha (arm D's 303456c1fa322b4c), so the key is optional.
    from NNNC._hold_window import HoldWindowMixin

    base = fp.load_strategy_class("SimpleTrigger_Gbb")
    held = type("HeldGbb", (HoldWindowMixin, base), {"HOLD_WINDOW_HOURS": 24.0})
    cfg = fp.load_config(CONFIG)
    sha_base, resolved_base = fp.resolve(base, None, cfg)
    sha_held, resolved_held = fp.resolve(held, None, cfg)
    assert resolved_held["attributes"]["HOLD_WINDOW_HOURS"] == 24.0
    assert "HOLD_WINDOW_HOURS" not in resolved_base["attributes"]
    assert sha_held != sha_base
    assert sha_base == "303456c1fa322b4c"
    assert _resolve("NNNC_A_Raw_Open")[0] == "303456c1fa322b4c"


# --- feature 024 T015 (contract C7, F1, F6): family members hashed only when used ------------


def test_shared_single_trigger_classes_keep_their_023_sha():
    for name in ("NNNC_A_Raw_Open", "SimpleTrigger_Gbb"):
        sha, resolved = _resolve(name)
        assert sha == "303456c1fa322b4c", name
        assert "FAMILY_EXITS_RESOLVED" not in resolved["attributes"], name


def test_family_mode_arm_hashes_its_family_sets():
    sha, resolved = _resolve("SimpleTrigger_FamBgo", params_json=None)
    attrs = resolved["attributes"]
    assert attrs["FAMILY_EXITS_RESOLVED"]["momentum"]["take_profit"] == 10.0
    assert attrs["FAMILY_EXITS_RESOLVED"]["momentum"]["unclog_12h_hours"] == 24.0
    assert attrs["FAMILY_EXITS_RESOLVED"]["reversion"]["take_profit"] == 0.022
    assert attrs["TRIGGER_FAMILIES"] == {"breakout_gbb": "momentum"}
    assert sha != "303456c1fa322b4c"


def _sha_of(cls):
    return fp.resolve(cls, None, fp.load_config(CONFIG))[0]


def test_each_family_dimension_moves_the_sha(monkeypatch):
    base = fp.load_strategy_class("SimpleTrigger_FamBgo")
    sha0 = _sha_of(base)
    # one momentum value
    tp = type("TP", (base,), {"FAMILY_EXIT_OVERRIDES": {"momentum": {"unclog_12h_hours": 20.0}}})
    assert _sha_of(tp) != sha0
    # the priority order (F6)
    pr = type("PR", (base,), {"FAMILY_PRIORITY": ("momentum", "reversion", "unassigned")})
    assert _sha_of(pr) != sha0
    # a trigger's family in the mapping (F6)
    from research.outcomes import triggers as registry

    registry.ensure_builtins()
    t = registry.get("breakout_gbb")
    import dataclasses

    monkeypatch.setitem(registry._REGISTRY, "breakout_gbb", dataclasses.replace(t, family="unassigned"))
    assert _sha_of(base) != sha0
