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
