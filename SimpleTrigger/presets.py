"""Framework presets: complete, named framework configurations (feature 023, data-model).

A preset is TOTAL. Every attribute the entry/exit/stop/sizing paths read, and every buy/sell
parameter value, is written here, so a SimpleTrigger arm inherits no framework value by accident.
A missing key is a load error (`validate`), never a silent inherited default.

The values were COPIED from the resolved reference classes on 2026-10-10 with
`python -m evaluation.fingerprint --strategy <ref> --json`, and
`SimpleTrigger/tests/test_presets.py` asserts each preset still resolves to its reference's
`framework_sha`. If a reference class changes, that test fails and this file must be re-copied
deliberately; it never follows silently.

  ARM_D_OPEN          the DEFAULT (spec FR-009). `NNNC_A_Raw_Open` + `NNNC_A_Raw_Open.json`: the
                      guards-open research operating point of arm D, the incumbent. Its twin needs
                      no overrides.
  LEGACY_TRIGGERONLY  `SimpleStrategies/TriggerOnly.TriggerOnlyStrategy`, verbatim. It exists ONLY
                      so SC-001 can reproduce the recorded 021/022 arms, which ran it. It is NOT
                      arm D's framework: its stop is a static -5% with the ATR-adaptive stop off
                      (which also makes its volume-confirmation stop inert), and its warm-up is
                      200 bars, not 64 (023 research R1).

`None` means "leave the attribute unset" unless the base class defines it, in which case it is
written as None (freqtrade treats an absent attribute and a None one differently).
"""

from __future__ import annotations

import copy


ARM_D_OPEN = {
    "attributes": {
        "MIN_QUOTE_VOLUME": 1000,
        "QUOTE_VOLUME_HEADROOM_MULT": 10.0,
        "atr_stoploss_cap": -0.04,
        "atr_stoploss_floor": -0.06,
        "atr_stoploss_multiplier": 4.0,
        "can_short": False,
        "entry_bear_deadband": 0.02,
        "entry_bear_ema_period": 200,
        "entry_bear_filter_enable": False,
        "entry_trend_ema_period": 100,
        "entry_trend_filter_enable": False,
        "exit_profit_only": True,
        "ignore_roi_if_entry_signal": True,
        "minimal_roi": {"0": 0.05},
        "position_adjustment_enable": False,
        "process_only_new_candles": True,
        "startup_candle_count": 64,
        "stoploss": -0.05,
        "timeframe": "15m",
        "trailing_only_offset_is_reached": False,
        "trailing_stop": False,
        "trailing_stop_positive": 0.01,
        "trailing_stop_positive_offset": 0.03,
        "use_atr_adaptive_stoploss": True,
        "use_custom_stoploss": True,
        "use_exit_signal": None,
        "use_volume_confirmation_stoploss": True,
    },
    "buy": {
        "entry_adx_threshold": -1.0,
        "entry_atr_pct": -1.0,
        "entry_bb_width_threshold": -1.0,
        "entry_close_norm_threshold": 5.0,
        "entry_enable_guards": True,
        "entry_guard_threshold": 5.0,
        "entry_rvol_threshold": -1.0,
    },
    "sell": {
        "cexit_enable_profit_checks": True,
        "cexit_max_days": 10,
        "cexit_take_profit": 0.022,
        "enable_exit_signal": False,
        "exit_close_norm_threshold": -0.6,
        "exit_guard_threshold": -0.1,
        "exit_rvol_threshold": 2.0,
        "stoploss_grace_hours": 6.5,
        "stoploss_grace_level": -0.27,
    },
}

LEGACY_TRIGGERONLY = {
    "attributes": {
        "MIN_QUOTE_VOLUME": 1000,
        "QUOTE_VOLUME_HEADROOM_MULT": 10.0,
        "atr_stoploss_cap": -0.02,
        "atr_stoploss_floor": -0.04,
        "atr_stoploss_multiplier": 2.5,
        "can_short": False,
        "entry_bear_deadband": 0.02,
        "entry_bear_ema_period": 200,
        "entry_bear_filter_enable": False,
        "entry_trend_ema_period": 200,
        "entry_trend_filter_enable": False,
        "exit_profit_only": True,
        "ignore_roi_if_entry_signal": True,
        "minimal_roi": {"0": 0.05},
        "position_adjustment_enable": False,
        "process_only_new_candles": True,
        "startup_candle_count": 200,
        "stoploss": -0.05,
        "timeframe": "15m",
        "trailing_only_offset_is_reached": False,
        "trailing_stop": False,
        "trailing_stop_positive": None,
        "trailing_stop_positive_offset": 0.0,
        "use_atr_adaptive_stoploss": False,
        "use_custom_stoploss": True,
        "use_exit_signal": True,
        "use_volume_confirmation_stoploss": True,
    },
    "buy": {
        "entry_adx_threshold": -1.0,
        "entry_atr_pct": -1.0,
        "entry_bb_width_threshold": -1.0,
        "entry_close_norm_threshold": 1.1,
        "entry_enable_guards": True,
        "entry_guard_threshold": 1.1,
        "entry_rvol_threshold": -1.0,
    },
    "sell": {
        "cexit_enable_profit_checks": True,
        "cexit_max_days": 10,
        "cexit_take_profit": 0.022,
        "enable_exit_signal": False,
        "exit_close_norm_threshold": 0.6,
        "exit_guard_threshold": -0.2,
        "exit_rvol_threshold": 2.0,
        "stoploss_grace_hours": 6.5,
        "stoploss_grace_level": -0.27,
    },
}

PRESETS = {"ARM_D_OPEN": ARM_D_OPEN, "LEGACY_TRIGGERONLY": LEGACY_TRIGGERONLY}

#: The buy/sell keys every preset must carry (the framework's hyperopt parameters, less the
#: excluded `prediction_threshold`).
BUY_KEYS = frozenset(ARM_D_OPEN["buy"])
SELL_KEYS = frozenset(ARM_D_OPEN["sell"])


class PresetError(ValueError):
    """A preset that is not total, or names a key the framework does not have."""


def validate(preset: dict) -> None:
    from evaluation.fingerprint import ATTRIBUTE_MEMBERS

    want = {"attributes": frozenset(ATTRIBUTE_MEMBERS), "buy": BUY_KEYS, "sell": SELL_KEYS}
    if set(preset) != set(want):
        raise PresetError(f"preset sections {sorted(preset)} != {sorted(want)}")
    for section, keys in want.items():
        have = set(preset[section])
        if have != keys:
            raise PresetError(
                f"preset[{section!r}] missing {sorted(keys - have)}, unknown {sorted(have - keys)}"
            )


def get(name: str) -> dict:
    """A validated deep copy of preset `name`."""
    if name not in PRESETS:
        raise PresetError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
    p = copy.deepcopy(PRESETS[name])
    validate(p)
    return p
