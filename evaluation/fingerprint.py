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
    "timeframe", "startup_candle_count", "process_only_new_candles", "can_short",
    # exits
    "minimal_roi", "use_exit_signal", "exit_profit_only", "ignore_roi_if_entry_signal",
    "position_adjustment_enable",
    # stop
    "stoploss", "use_custom_stoploss", "trailing_stop", "trailing_stop_positive",
    "trailing_stop_positive_offset", "trailing_only_offset_is_reached",
    "use_atr_adaptive_stoploss", "atr_stoploss_multiplier", "atr_stoploss_floor",
    "atr_stoploss_cap", "use_volume_confirmation_stoploss",
    # entry gates outside the guard thresholds
    "entry_bear_filter_enable", "entry_bear_ema_period", "entry_bear_deadband",
    "entry_trend_filter_enable", "entry_trend_ema_period",
    # sizing / liquidity
    "MIN_QUOTE_VOLUME", "QUOTE_VOLUME_HEADROOM_MULT",
)

#: Hyperopt parameters excluded from the hash: they shape the entry SIGNAL, which a twin may
#: change, not the framework around it.
EXCLUDED_PARAMS = frozenset({"prediction_threshold"})

#: JSON sections that REPLACE an attribute outright (hyper.py `load_params_from_file` /
#: `ft_bot_start` special parameters).
_TRAILING_KEYS = ("trailing_stop", "trailing_stop_positive", "trailing_stop_positive_offset",
                  "trailing_only_offset_is_reached")

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


def resolve(strategy_cls, params_json: dict | None, config: dict) -> tuple[str, dict]:
    """(framework_sha, resolved_dict) for a strategy class + its params JSON + run config."""
    file_params = (params_json or {}).get("params", {}) or {}

    attributes = {}
    for key in ATTRIBUTE_MEMBERS:
        attributes[key] = _jsonable(getattr(strategy_cls, key, None))
    # JSON special sections replace the attribute (hyper.py ft_bot_start / load_params)
    if "roi" in file_params:
        attributes["minimal_roi"] = _jsonable(file_params["roi"])
    sl = file_params.get("stoploss")
    if isinstance(sl, dict) and "stoploss" in sl:
        attributes["stoploss"] = sl["stoploss"]
    for k in _TRAILING_KEYS:
        if k in file_params.get("trailing", {}):
            attributes[k] = file_params["trailing"][k]

    spaces = {}
    params = _parameters(strategy_cls)
    for space in ("buy", "sell"):
        merged = _deep_merge(file_params.get(space, {}) or {},
                             getattr(strategy_cls, f"{space}_params", {}) or {})
        resolved_space = {}
        for name, param in params[space].items():
            if name in EXCLUDED_PARAMS:
                continue
            if name in merged and getattr(param, "load", True):
                resolved_space[name] = merged[name]
            else:
                # .value is class-shared and may have been mutated by an earlier instance in
                # this process; the default is what a fresh backtest starts from.
                resolved_space[name] = getattr(param, "default", param.value)
        # keys present in the dict without a Parameter object are still read by code
        for name, v in merged.items():
            if name not in resolved_space and name not in EXCLUDED_PARAMS:
                resolved_space[name] = v
        spaces[space] = {k: _jsonable(v) for k, v in sorted(resolved_space.items())}

    mot = file_params.get("max_open_trades", {})
    max_open = mot.get("max_open_trades") if isinstance(mot, dict) else None
    if max_open is None:
        max_open = config.get("max_open_trades", getattr(strategy_cls, "max_open_trades", None))
    run = {"max_open_trades": max_open}
    for k in RUN_CONFIG_KEYS:
        run[k] = _jsonable(config.get(k))

    resolved = {"attributes": attributes, "buy": spaces["buy"], "sell": spaces["sell"],
                "run": run}
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
