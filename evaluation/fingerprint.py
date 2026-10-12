"""Resolved-framework fingerprint (feature 023, research R4, contract C3/C5).

WHY THIS EXISTS. `regime/run_arm.py`'s `exit_config_hash` hashes only the sell/roi/stoploss/
trailing sections of a strategy's params JSON. A strategy whose exits live in class attributes
-- every 021 trigger-only arm -- has no JSON, so all of them hashed to the same value whatever
their real exit was. Equal hashes there meant "no JSON overrides", not "same exit". That is how
TriggerOnly's static stop went unnoticed next to arm D's ATR-adaptive one.

This module hashes what freqtrade will actually RUN: the class attributes the entry, exit and
stop paths read (resolved across the MRO), every buy/sell hyperopt parameter's effective value
(class dict and params JSON merged exactly as `freqtrade/strategy/hyper.py` merges them, falling
back to the parameter default), and the run config's sizing. Two arms with the same
`framework_sha` share a framework; that is the precondition for a twin comparison (FR-013).

EXCLUDED, deliberately: the entry-signal source and everything that only shapes it --
`prediction_threshold`, the classifier, label parameters. Those are exactly what a no-model twin
is allowed to differ in.

NEVER INSTANTIATES. Instantiating an NNNC strategy loads a model or, with none on disk, trains
one inside the backtest (`get_model_path` keys off the class name). Everything here reads class
objects only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


STRAT_DIR = Path(__file__).resolve().parents[1]
ROOT = STRAT_DIR.parents[1]

#: Class attributes the framework reads (BaseStrategy entry/exit/stop/sizing paths, plus the
#: freqtrade-level exit switches). data-model `FrameworkPreset`.
ATTRIBUTE_MEMBERS = (
    # run shape
    "timeframe",
    "startup_candle_count",
    "process_only_new_candles",
    "can_short",
    # exits
    "minimal_roi",
    "use_exit_signal",
    "exit_profit_only",
    "ignore_roi_if_entry_signal",
    "position_adjustment_enable",
    # stop
    "stoploss",
    "use_custom_stoploss",
    "trailing_stop",
    "trailing_stop_positive",
    "trailing_stop_positive_offset",
    "trailing_only_offset_is_reached",
    "use_atr_adaptive_stoploss",
    "atr_stoploss_multiplier",
    "atr_stoploss_floor",
    "atr_stoploss_cap",
    "use_volume_confirmation_stoploss",
    # entry gates outside the guard thresholds
    "entry_bear_filter_enable",
    "entry_bear_ema_period",
    "entry_bear_deadband",
    "entry_trend_filter_enable",
    "entry_trend_ema_period",
    # sizing / liquidity
    "MIN_QUOTE_VOLUME",
    "QUOTE_VOLUME_HEADROOM_MULT",
)

#: Attributes hashed ONLY when set (not None). Mixin knobs that change exits but are absent from
#: most strategies: listing them in ATTRIBUTE_MEMBERS would add a `None` key to every resolved
#: dict and change every recorded framework_sha. HOLD_WINDOW_HOURS (NNNC/_hold_window.py) defers
#: unclog_12h; before this, an arm with it hashed identically to the same arm without it.
OPTIONAL_ATTRIBUTE_MEMBERS = ("HOLD_WINDOW_HOURS", "take_profit_atr_mult")

#: Default-off ExitParams switches added after recorded arms were hashed (028 C2). Omitted from
#: the resolved family sets while None, so every recorded `framework_sha` is unchanged (FR-010);
#: switched on, they are hashed like any other field.
SWITCH_FIELDS_OMITTED_WHEN_OFF = frozenset({"take_profit_atr_mult"})

#: Hyperopt parameters excluded from the hash: they shape the entry SIGNAL, which a twin may
#: change, not the framework around it.
EXCLUDED_PARAMS = frozenset({"prediction_threshold"})

#: JSON sections that REPLACE an attribute outright (hyper.py `load_params_from_file` /
#: `ft_bot_start` special parameters).
_TRAILING_KEYS = (
    "trailing_stop",
    "trailing_stop_positive",
    "trailing_stop_positive_offset",
    "trailing_only_offset_is_reached",
)

RUN_CONFIG_KEYS = ("stake_amount", "dry_run_wallet", "fee")


def _ensure_paths() -> None:
    for p in (str(STRAT_DIR), str(STRAT_DIR / "utils")):
        if p not in sys.path:
            sys.path.insert(0, p)


def strategy_file(name: str) -> Path:
    """`<group dir>/<name>.py`, resolved with run_arm's own lookup so both agree."""
    _ensure_paths()
    from regime.run_arm import strategy_group  # local: keeps import of this module light

    return STRAT_DIR / strategy_group(name) / f"{name}.py"


def load_strategy_class(name: str):
    """Import strategy class `name` from its file. Does NOT instantiate it."""
    path = strategy_file(name)
    if not path.is_file():
        raise FileNotFoundError(f"no strategy file for {name!r} at {path}")
    _ensure_paths()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    mod_name = f"_fp_{path.parent.name}_{name}"
    if mod_name in sys.modules:
        return getattr(sys.modules[mod_name], name)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def load_params_json(name: str) -> dict | None:
    """The strategy's `<name>.json` beside its file, or None when it has none."""
    js = strategy_file(name).with_suffix(".json")
    return json.loads(js.read_text()) if js.is_file() else None


def load_config(path: str | Path) -> dict:
    """A freqtrade config, read with freqtrade's own loader (configs carry // comments)."""
    from freqtrade.configuration.load_config import load_config_file

    return load_config_file(str(path))


def _deep_merge(source: dict, destination: dict) -> dict:
    """freqtrade.misc.deep_merge_dicts semantics: SOURCE wins, nested dicts merge."""
    out = copy.deepcopy(destination)
    for k, v in source.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(v, out[k])
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parameters(cls) -> dict[str, dict]:
    """{space: {name: BaseParameter}} for buy/sell, detected the way hyper.py detects them."""
    from freqtrade.strategy.parameters import BaseParameter

    found: dict[str, dict] = {"buy": {}, "sell": {}}
    for attr_name in dir(cls):
        if attr_name.startswith("__"):
            continue
        attr = getattr(cls, attr_name, None)
        if not isinstance(attr, BaseParameter):
            continue
        space = attr.space
        if not space:
            space = next((s for s in ("buy", "sell") if attr_name.startswith(s + "_")), None)
        if space in found:
            found[space][attr_name] = attr
    return found


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


def _family_members(strategy_cls, sell_values: dict) -> dict:
    """Feature 024 C7 / F1: trigger-family members, ONLY for an arm that uses them.

    Included when the class is in family mode (some family exit set non-empty), multi-trigger
    mode (more than one trigger) or disabled-side mode (non-empty DISABLED_TRIGGERS). A
    single-trigger, shared-exit class -- every 023 config -- hashes exactly as before, so
    SimpleTrigger_Gbb stays arm D's twin (303456c1fa322b4c).
    """
    sets = getattr(strategy_cls, "FAMILY_EXITS", None) or {}
    triggers = tuple(getattr(strategy_cls, "TRIGGERS", None) or ())
    if not triggers and getattr(strategy_cls, "TRIGGER_NAME", None):
        triggers = (strategy_cls.TRIGGER_NAME,)
    disabled = tuple(getattr(strategy_cls, "DISABLED_TRIGGERS", None) or ())
    # 025 C4: coexistence rules, hashed only when non-neutral (so 024 shas are unchanged)
    coexist = {}
    if getattr(strategy_cls, "FAMILY_PAIR_CAP", None) is not None:
        coexist["FAMILY_PAIR_CAP"] = strategy_cls.FAMILY_PAIR_CAP
    if (getattr(strategy_cls, "MOMENTUM_RECENCY_BARS", 0) or 0) > 0:
        coexist["MOMENTUM_RECENCY_BARS"] = strategy_cls.MOMENTUM_RECENCY_BARS
    if getattr(strategy_cls, "MOMENTUM_PREEMPT", False):  # 026 addendum 2
        coexist["MOMENTUM_PREEMPT"] = True
    if getattr(strategy_cls, "REVERSION_REGIME_GATE", None) is not None:  # 026 C4
        coexist["REVERSION_REGIME_GATE"] = _jsonable(strategy_cls.REVERSION_REGIME_GATE)
    # 029: the entry source. The trigger gate on a model is a switch, so it forces the block; a
    # model's entry family is hashed whenever the block exists. Trigger mode and plain model mode
    # add nothing, so every pre-029 sha is unchanged.
    mode = getattr(strategy_cls, "ENTRY_MODE", None)
    if mode == "trigger_and_model":
        coexist["ENTRY_MODE"] = mode
    if not (any(sets.values()) or len(triggers) > 1 or disabled or coexist):
        return {}
    entry = {}
    if mode in ("model", "trigger_and_model"):
        entry = {"ENTRY_MODE": mode, "MODEL_ENTRY_FAMILY": strategy_cls._model_family()}
    import dataclasses

    from research.outcomes import triggers as registry
    from utils.FamilyExits import FamilyExitMixin, class_level_exit_params

    registry.ensure_builtins()
    if not issubclass(strategy_cls, FamilyExitMixin):
        raise TypeError(f"{strategy_cls.__name__}: family members without FamilyExitMixin")
    # The resolved sell space (JSON merged), not the Parameters' class-level defaults.
    base = class_level_exit_params(strategy_cls, sell_values)
    return {
        "TRIGGERS": list(triggers),
        "DISABLED_TRIGGERS": list(disabled),
        "FAMILY_PRIORITY": list(getattr(strategy_cls, "FAMILY_PRIORITY", None) or ()),
        "FAMILY_EXITS_RESOLVED": {
            family: _jsonable(_without_off_switches(
                dataclasses.asdict(dataclasses.replace(base, **(overrides or {})))))
            for family, overrides in sorted(sets.items())
        },
        "TRIGGER_FAMILIES": {t: registry.get(t).family for t in triggers},
        **coexist,
        **entry,
    }


def _without_off_switches(fields: dict) -> dict:
    return {k: v for k, v in fields.items()
            if not (k in SWITCH_FIELDS_OMITTED_WHEN_OFF and v is None)}


def _resolve_attributes(strategy_cls, file_params: dict) -> dict:
    attributes = {key: _jsonable(getattr(strategy_cls, key, None)) for key in ATTRIBUTE_MEMBERS}
    for key in OPTIONAL_ATTRIBUTE_MEMBERS:
        value = getattr(strategy_cls, key, None)
        if value is not None:
            attributes[key] = _jsonable(value)

    # JSON special sections replace the attribute (hyper.py ft_bot_start / load_params)
    if "roi" in file_params:
        attributes["minimal_roi"] = _jsonable(file_params["roi"])
    sl = file_params.get("stoploss")
    if isinstance(sl, dict) and "stoploss" in sl:
        attributes["stoploss"] = sl["stoploss"]
    trailing = file_params.get("trailing", {})
    for k in _TRAILING_KEYS:
        if k in trailing:
            attributes[k] = trailing[k]
    return attributes


def _resolve_space(strategy_cls, params: dict, file_params: dict, space: str) -> dict:
    merged = _deep_merge(
        file_params.get(space, {}) or {}, getattr(strategy_cls, f"{space}_params", {}) or {}
    )
    out = {}
    for name, param in params[space].items():
        if name in EXCLUDED_PARAMS:
            continue
        if name in merged and getattr(param, "load", True):
            out[name] = merged[name]
        else:
            # .value is class-shared and may have been mutated by an earlier instance in
            # this process; the default is what a fresh backtest starts from.
            out[name] = getattr(param, "default", param.value)
    # keys present in the dict without a Parameter object are still read by code
    for name, v in merged.items():
        if name not in out and name not in EXCLUDED_PARAMS:
            out[name] = v
    return {k: _jsonable(v) for k, v in sorted(out.items())}


def resolve(strategy_cls, params_json: dict | None, config: dict) -> tuple[str, dict]:
    """(framework_sha, resolved_dict) for a strategy class + its params JSON + run config."""
    file_params = (params_json or {}).get("params", {}) or {}
    params = _parameters(strategy_cls)
    mot = file_params.get("max_open_trades", {})
    max_open = mot.get("max_open_trades") if isinstance(mot, dict) else None
    if max_open is None:
        max_open = config.get("max_open_trades", getattr(strategy_cls, "max_open_trades", None))
    run = {"max_open_trades": max_open, **{k: _jsonable(config.get(k)) for k in RUN_CONFIG_KEYS}}
    resolved = {
        "attributes": _resolve_attributes(strategy_cls, file_params),
        "buy": _resolve_space(strategy_cls, params, file_params, "buy"),
        "sell": _resolve_space(strategy_cls, params, file_params, "sell"),
        "run": run,
    }
    resolved["attributes"].update(_family_members(strategy_cls, resolved["sell"]))
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16], resolved


def resolve_by_name(name: str, config_path: str | Path) -> tuple[str, dict]:
    return resolve(load_strategy_class(name), load_params_json(name), load_config(config_path))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--config", default=str(STRAT_DIR / "config" / "config_wf.json"))
    ap.add_argument("--json", action="store_true", help="print the full resolved dict")
    a = ap.parse_args(argv)
    sha, resolved = resolve_by_name(a.strategy, a.config)
    print(f"{a.strategy}  framework_sha {sha}")
    if a.json:
        print(json.dumps(resolved, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
