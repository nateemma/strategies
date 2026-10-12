# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error
"""SimpleTrigger: NNNC's framework with no model (feature 023, contract C1).

An NNNC strategy is a MODEL that gives an entry opinion, inside a FRAMEWORK: the guard chain,
the gain/loss targets, the exits and the custom stop. Every NNNC result mixes the two. A
SimpleTrigger strategy is the framework alone -- its entry is a registered, measured-causal
trigger fed through `BaseStrategy`'s unchanged guard chain -- so the framework can be evaluated
on its own, and an NNNC arm can be compared against a twin that differs ONLY in the model.

A configuration is a subclass that names a trigger and, optionally, a preset and overrides:

    class SimpleTrigger_Gbb(SimpleTriggerStrategy):
        TRIGGER_NAME = "gbb"

GUARANTEES (contract C1)
  G1 no model: StrategyConfig model NONE, needs_training False. Nothing is loaded, trained or
     saved, so the renamed-strategy self-training hazard cannot occur.
  G2 the entry is `research.outcomes.triggers.evaluate(TRIGGER_NAME, df)` -- the registry's own
     definition, never a copy -- and only for a trigger that PASSES the measured causality check
     with the current code (`research/outcomes/trigger_causality.json`). Otherwise the strategy
     enters NO trades and says why; there is no fallback condition.
  G3 the framework is exactly the named PRESET (`SimpleTrigger/presets.py`, default ARM_D_OPEN =
     arm D), plus explicit BUY_/SELL_/ATTR_OVERRIDES. Parameter dicts are MERGED, never redefined:
     a subclass that declares its own `buy_params`/`sell_params` is refused.
  G4 `custom_stoploss` runs through `StopInvariantMixin`: a wrong-sign, over-loose or
     wrong-direction stop halts the run (FR-017).
  G6 no pair-specific behaviour.

NO REGIME LOGIC (FR-005). Unlike `SimpleStrategies.SimpleStrategy` there is no bear/bull/sideways
filter. `exit_profit_only` is whatever the PRESET says, because parity with NNNC wins: arm D
inherits `exit_profit_only = True` from BaseStrategy, so ARM_D_OPEN carries it too.

ENTRY LAG. A signal on the last closed candle fills at the next candle's open -- the same pairing
the outcome table uses (`labels.entry_mask`). No manual shift; adding one would double-lag.

ENTRY TAG AND FAMILIES (feature 024, contract C4). Every signal row is tagged
`"<family>:<trigger>"` (family from the trigger registry), replacing BaseStrategy's
"model_entry ". freqtrade copies the tag onto the trade, and `FamilyExitMixin` chooses that
trade's exit parameters from it for the trade's whole life. Exits are per-family only under a
preset that carries family sets (`ARM_D_FAMILY`); under `ARM_D_OPEN` / `LEGACY_TRIGGERONLY`
every family resolves to the shared exits, so 023's configs are unchanged. Nothing reads the
tag except the exit hook, and `trades_hash` does not include it.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame


sys.path.append(str(Path(__file__).parent.parent))

from Framework.BaseStrategy import BaseStrategy, StrategyConfig
from SimpleTrigger import presets as _presets
from utils.FamilyCoexistence import FamilyCoexistenceMixin
from utils.FamilyExits import FamilyExitMixin, family_of, validate_family_exits
from utils.StopInvariants import StopInvariantMixin


EXIT_MODES = ("decoupled", "coupled")


def _apply_preset(cls) -> None:
    """Write the preset (+ overrides) onto `cls`. Called for the base and every subclass."""
    for forbidden in ("buy_params", "sell_params", "FAMILY_EXITS"):
        if forbidden in cls.__dict__ and cls is not SimpleTriggerStrategy:
            raise _presets.PresetError(
                f"{cls.__name__} declares its own `{forbidden}`. Use BUY_OVERRIDES / "
                f"SELL_OVERRIDES / FAMILY_EXIT_OVERRIDES: a fresh dict silently drops "
                f"inherited keys (constitution)."
            )
    if cls.EXIT_MODE not in EXIT_MODES:
        raise _presets.PresetError(
            f"{cls.__name__}.EXIT_MODE {cls.EXIT_MODE!r} not in {EXIT_MODES}"
        )
    preset = _presets.get(cls.PRESET)
    attrs = {**preset["attributes"], **cls.ATTR_OVERRIDES}
    buy = {**preset["buy"], **cls.BUY_OVERRIDES}
    sell = {**preset["sell"], **cls.SELL_OVERRIDES}

    if cls.EXIT_MODE == "coupled":
        from SimpleTrigger.coupling import resolve as _resolve_coupling

        res = _resolve_coupling(cls)
        sell.update(res["sell"])
        attrs.update(res["attributes"])
        cls.COUPLED_RESOLUTION = res["record"]
    else:
        cls.COUPLED_RESOLUTION = None

    for k, v in attrs.items():
        if v is None and not hasattr(BaseStrategy, k):
            if getattr(cls, k, None) is not None:
                raise _presets.PresetError(
                    f"{cls.__name__}: preset leaves {k!r} unset but a parent sets it to "
                    f"{getattr(cls, k)!r}; an inherited value cannot be unset"
                )
            continue
        setattr(cls, k, v)
    cls.buy_params = {**BaseStrategy.buy_params, **buy}
    cls.sell_params = {**BaseStrategy.sell_params, **sell}

    # 024: per-family exit sets = preset's, merged PER FAMILY with FAMILY_EXIT_OVERRIDES.
    # 025 G1: overrides MERGE along the MRO (ancestors first, per family, per key), so a subclass
    # adds to its parent's overrides instead of silently replacing them.
    families = _presets.family_exits(preset)
    merged: dict = {}
    for klass in reversed(cls.__mro__):
        for family, overrides in (klass.__dict__.get("FAMILY_EXIT_OVERRIDES") or {}).items():
            merged.setdefault(family, {}).update(overrides)
    unknown = sorted(set(merged) - set(families))
    if unknown:
        raise _presets.PresetError(
            f"{cls.__name__}: FAMILY_EXIT_OVERRIDES unknown family {unknown}"
        )
    for family, overrides in merged.items():
        families[family] = {**families[family], **overrides}
    cls.FAMILY_EXITS = families
    _validate_triggers(cls)


#: 026: a market-wide regime instrument needs ~2,900 bars of history to be defined from a fold's
#: first bar (research R4/H1); a power of 2 per the BaseStrategy startup comment.
REGIME_GATE_MIN_STARTUP = 4096


def _validate_regime_gate(cls, names) -> None:
    gate = cls.REVERSION_REGIME_GATE
    if gate is None:
        return
    from utils import RegimeState as _rs

    if not isinstance(gate, dict) or gate.get("instrument") not in _rs.INSTRUMENTS:
        raise _presets.PresetError(
            f"{cls.__name__}: REVERSION_REGIME_GATE instrument must be one of "
            f"{sorted(_rs.INSTRUMENTS)}, got {gate!r}"
        )
    if gate.get("off_when") not in ("ge", "lt"):
        raise _presets.PresetError(f"{cls.__name__}: REVERSION_REGIME_GATE off_when must be ge/lt")
    thr = gate.get("threshold")
    if not isinstance(thr, (int, float)) or not np.isfinite(thr):
        raise _presets.PresetError(
            f"{cls.__name__}: REVERSION_REGIME_GATE threshold must be finite"
        )
    if int(cls.startup_candle_count) < REGIME_GATE_MIN_STARTUP:
        raise _presets.PresetError(
            f"{cls.__name__}: REVERSION_REGIME_GATE needs startup_candle_count >= "
            f"{REGIME_GATE_MIN_STARTUP} (got {cls.startup_candle_count}); a shorter warm-up leaves "
            f"the instrument undefined for part of each fold (026 H1)"
        )
    from research.outcomes import triggers as _triggers

    _triggers.ensure_builtins()
    enabled = [t for t in names if t not in cls.DISABLED_TRIGGERS]
    if not any(_triggers.get(t).family == "reversion" for t in enabled):
        raise _presets.PresetError(
            f"{cls.__name__}: REVERSION_REGIME_GATE needs an enabled reversion trigger; "
            f"without one it is INERT by construction"
        )


def _validate_triggers(cls) -> None:
    """024 C4: the trigger declaration is unambiguous and the priority is a full order."""
    if cls.TRIGGER_NAME and cls.TRIGGERS:
        raise _presets.PresetError(
            f"{cls.__name__}: set TRIGGER_NAME or TRIGGERS, not both"
        )
    names = tuple(cls.TRIGGERS or ((cls.TRIGGER_NAME,) if cls.TRIGGER_NAME else ()))
    stray = sorted(set(cls.DISABLED_TRIGGERS) - set(names))
    if stray:
        raise _presets.PresetError(f"{cls.__name__}: DISABLED_TRIGGERS {stray} not in TRIGGERS")
    if names and set(names) <= set(cls.DISABLED_TRIGGERS):
        raise _presets.PresetError(f"{cls.__name__}: DISABLED_TRIGGERS disables every trigger")
    if sorted(cls.FAMILY_PRIORITY) != sorted(_presets.FAMILIES):
        raise _presets.PresetError(
            f"{cls.__name__}: FAMILY_PRIORITY {cls.FAMILY_PRIORITY} is not a permutation of "
            f"{_presets.FAMILIES}"
        )
    _validate_regime_gate(cls, names)
    window = cls.MOMENTUM_RECENCY_BARS
    if type(window) is not int or window < 0:
        raise _presets.PresetError(
            f"{cls.__name__}: MOMENTUM_RECENCY_BARS must be an int >= 0, got {window!r}"
        )
    if window > 0 and cls.MOMENTUM_PREEMPT:
        raise _presets.PresetError(
            f"{cls.__name__}: MOMENTUM_RECENCY_BARS and MOMENTUM_PREEMPT are alternative "
            f"momentum-precedence rules; set one"
        )
    for rule, on in (("MOMENTUM_RECENCY_BARS", window > 0),
                     ("MOMENTUM_PREEMPT", bool(cls.MOMENTUM_PREEMPT))):
        if not on:
            continue
        from research.outcomes import triggers as _triggers

        _triggers.ensure_builtins()
        enabled = [t for t in names if t not in cls.DISABLED_TRIGGERS]
        if not any(_triggers.get(t).family == "momentum" for t in enabled):
            raise _presets.PresetError(
                f"{cls.__name__}: {rule} needs an enabled momentum "
                f"trigger; without one the rule is INERT by construction"
            )


class SimpleTriggerStrategy(
    FamilyCoexistenceMixin, FamilyExitMixin, StopInvariantMixin, BaseStrategy
):
    """Trigger + framework preset, no model. Subclass and set TRIGGER_NAME."""

    #: 025 rule B: drop a reversion-only entry if a momentum-family trigger fired on the pair
    #: within the last N bars (current included); while active, overlap bars go to momentum.
    #: 0 = off (024 behaviour). Rule A (FAMILY_PAIR_CAP) comes from FamilyCoexistenceMixin.
    MOMENTUM_RECENCY_BARS: int = 0
    #: 026 addendum 2: momentum never waits for a pair reversion holds. A fillable momentum signal
    #: on a pair held by a reversion trade exits that trade at the next open ("momentum_preempt");
    #: momentum enters on the following bar if its signal is still live (freqtrade evaluates a
    #: bar's entry before its exits). No late copy of the signal: an extra entry signal suppresses
    #: ROI on any open trade (ignore_roi_if_entry_signal) and broke the null check. Overlap bars
    #: go to momentum. False = off.
    MOMENTUM_PREEMPT: bool = False
    #: 026: None, or {"instrument": <utils.RegimeState id>, "threshold": float,
    #: "off_when": "ge"|"lt"}. In the "off" state reversion-only entries are dropped and overlap
    #: entries go to momentum; NaN state = "on". Requires a >= 4096-bar startup.
    REVERSION_REGIME_GATE: dict | None = None

    #: Combined arm (024 C4): several triggers OR'ed. Exactly one of TRIGGER_NAME / TRIGGERS.
    TRIGGERS: tuple | None = None
    #: Triggers not evaluated at all -- disabling one side reproduces the other exactly (FR-009).
    DISABLED_TRIGGERS: tuple = ()
    #: On a bar where triggers of several families fire, the earliest family here wins the tag.
    FAMILY_PRIORITY: tuple = ("reversion", "momentum", "unassigned")

    # FAMILY_EXITS is resolved by `_apply_preset`, so validate after it, not on class creation.
    _FAMILY_VALIDATE_ON_SUBCLASS = False
    #: {family: {ExitParams field: value}} merged over the preset's family set (024).
    FAMILY_EXIT_OVERRIDES: dict = {}

    TRIGGER_NAME: str | None = None
    PRESET: str = "ARM_D_OPEN"
    EXIT_MODE: str = "decoupled"
    LABEL_METHOD = None  # required iff EXIT_MODE == "coupled"
    BUY_OVERRIDES: dict = {}
    SELL_OVERRIDES: dict = {}
    ATTR_OVERRIDES: dict = {}
    COUPLED_RESOLUTION = None

    # G1. Explicit rather than inherited, so the guarantee is visible where it is relied on.
    strategy_config = StrategyConfig()

    def __init_subclass__(cls, **kwargs):
        _apply_preset(cls)
        super().__init_subclass__(**kwargs)
        validate_family_exits(cls)

    def _refuse(self, reason: str, n: int) -> np.ndarray:
        if not getattr(self, "_st_refusal_reported", False):
            self._st_refusal_reported = True
            print(
                f"*** SimpleTrigger: {self.__class__.__name__} enters NO trades -- {reason}",
                flush=True,
            )
        return np.zeros(n, dtype=bool)

    def _enabled_triggers(self) -> tuple:
        names = tuple(self.TRIGGERS or ((self.TRIGGER_NAME,) if self.TRIGGER_NAME else ()))
        return tuple(n for n in names if n not in self.DISABLED_TRIGGERS)

    def get_entry_conditions(self, dataframe: DataFrame):
        n = len(dataframe)
        names = self._enabled_triggers()
        if not names:
            return self._refuse(
                "TRIGGER_NAME is not set; a trigger-less arm would silently "
                "measure the guards instead",
                n,
            )
        if getattr(self, "_st_runnable", None) is None:
            try:
                from research.outcomes.causality import check_runnable

                self._st_runnable = {name: check_runnable(name) for name in names}
            except Exception as exc:
                why = f"causality check unavailable ({type(exc).__name__}: {exc})"
                self._st_runnable = {name: (False, why) for name in names}
        for name in names:  # ANY unrunnable trigger refuses the whole arm: no silent one-sided run
            ok, why = self._st_runnable[name]
            if not ok:
                return self._refuse(f"trigger {name!r} is not runnable: {why}", n)
        try:
            from research.outcomes import triggers as _triggers

            _triggers.ensure_builtins()
            masks = [_triggers.evaluate(name, dataframe).to_numpy() for name in names]
            families = [_triggers.get(name).family for name in names]
        except Exception as exc:
            return self._refuse(
                f"could not evaluate trigger(s) {names} ({type(exc).__name__}: "
                f"{exc}); no fallback definition is used",
                n,
            )
        window = self.MOMENTUM_RECENCY_BARS
        priority = self.FAMILY_PRIORITY
        if window > 0 or self.MOMENTUM_PREEMPT:  # 025 G2: overlap bars go to momentum
            priority = ("momentum",) + tuple(f for f in priority if f != "momentum")
        tags = self._tags(names, masks, families, n, priority)
        mask = np.logical_or.reduce(masks)
        self._st_coexist_refused = 0
        if window > 0:
            mom = np.logical_or.reduce(
                [m for m, f in zip(masks, families, strict=True) if f == "momentum"]
            )
            recent = pd.Series(mom).rolling(window, min_periods=1).max().to_numpy() > 0
            winner = np.array([t.split(":", 1)[0] for t in tags], dtype=object)
            drop = (winner == "reversion") & recent
            tags[drop] = ""
            mask = mask & ~drop
            self._st_coexist_refused = int(drop.sum())
        if self.REVERSION_REGIME_GATE is not None:
            tags, mask = self._apply_regime_gate(dataframe, names, masks, families, tags, mask)
        self._st_tags = tags
        return mask

    # ------------------------------------------------------------- 026 momentum preempt

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        if self.MOMENTUM_PREEMPT and family_of(getattr(trade, "enter_tag", None)) == "reversion":
            dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            last = dataframe.iloc[-1]
            tag = last.get("enter_tag")
            # Only for a FILLABLE signal: the same quote-volume floor confirm_trade_entry applies.
            # A thin signal bar would not have been entered by momentum alone either (smoke f00:
            # 27/66 handovers lost to a zero stake), so preempting for it only cuts reversion.
            fillable = last["volume"] * last["close"] >= self.MIN_QUOTE_VOLUME
            if fillable and last.get("enter_long") == 1 and family_of(tag) == "momentum":
                return "momentum_preempt"
        return super().custom_exit(pair, trade, current_time, current_rate, current_profit,
                                   **kwargs)

    # ------------------------------------------------------------------ 026 regime gate

    def informative_pairs(self):
        gate = self.REVERSION_REGIME_GATE
        if gate is not None and gate["instrument"] == "btctrend":
            return [(_presets.REGIME_CONTEXT_PAIR, self.timeframe)]
        return []

    def _regime_state(self) -> pd.Series:
        """The gate's instrument over the whitelist (and BTC), cached per latest bar."""
        from utils import RegimeState as _rs

        gate = self.REVERSION_REGIME_GATE
        frames = {p: self.dp.get_pair_dataframe(p, self.timeframe)
                  for p in self.dp.current_whitelist()}
        btc = (self.dp.get_pair_dataframe(_presets.REGIME_CONTEXT_PAIR, self.timeframe)
               if gate["instrument"] == "btctrend" else None)
        last = max((f["date"].iloc[-1] for f in frames.values() if len(f)), default=None)
        key = (gate["instrument"], last, len(frames))
        cache = getattr(self, "_st_regime_cache", None)
        if cache is None or cache[0] != key:
            cache = (key, _rs.compute(gate["instrument"], frames, btc).sort_index())
            self._st_regime_cache = cache
        return cache[1]

    def _apply_regime_gate(self, dataframe, names, masks, families, tags, mask):
        gate = self.REVERSION_REGIME_GATE
        n = len(dataframe)
        state = self._regime_state().reindex(pd.DatetimeIndex(dataframe["date"]), method="ffill")
        vals = state.to_numpy(dtype=float)
        thr = gate["threshold"]
        off = (vals >= thr) if gate["off_when"] == "ge" else (vals < thr)
        off = off & ~np.isnan(vals)
        winner = np.array([t.split(":", 1)[0] if t else "" for t in tags], dtype=object)
        mom = [i for i, f in enumerate(families) if f == "momentum"]
        mom_any = np.logical_or.reduce([masks[i] for i in mom]) if mom else np.zeros(n, bool)
        hit = (winner == "reversion") & off
        retag = hit & mom_any
        drop = hit & ~mom_any
        if retag.any():
            first = np.full(n, "", dtype=object)
            for i in reversed(mom):
                first[masks[i]] = names[i]
            tags[retag] = np.array([f"momentum:{t}+ov" for t in first[retag]], dtype=object)
        tags[drop] = ""
        self._st_regime_counts = {
            "regime_gate_refused": int(drop.sum()),
            "regime_gate_retagged": int(retag.sum()),
            "regime_gate_off_bars": int(off.sum()),
        }
        return tags, mask & ~drop

    def _tags(self, names, masks, families, n, priority=None) -> np.ndarray:
        """`<family>:<trigger>` per signal row, by family priority then trigger order; `+ov` when
        a trigger of ANOTHER family also fired on that row (024 C4, F4)."""
        rank = {f: i for i, f in enumerate(priority or self.FAMILY_PRIORITY)}
        order = sorted(range(len(names)), key=lambda i: (rank.get(families[i], len(rank)), i))
        tags = np.full(n, "", dtype=object)
        winner = np.full(n, "", dtype=object)
        for i in order:
            take = masks[i] & (tags == "")
            if families[i]:
                tags[take] = f"{families[i]}:{names[i]}"
                winner[take] = families[i]
        overlap = np.zeros(n, dtype=bool)
        for i in range(len(names)):
            overlap |= masks[i] & (winner != "") & (winner != (families[i] or ""))
        tags[overlap] = tags[overlap] + "+ov"
        return tags


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """BaseStrategy's entry, then the family tag on every signal row (024 C4)."""
        self._st_tags = None
        dataframe = super().populate_entry_trend(dataframe, metadata)
        tags = self._st_tags
        if tags is not None and len(tags) == len(dataframe):
            signal = tags != ""
            dataframe.loc[signal, "enter_tag"] = tags[signal]
        self._sink_coexistence(metadata.get("pair", "?"), len(dataframe))
        return dataframe

    def _sink_coexistence(self, pair: str, rows: int) -> None:
        """025 C3: rule B's dropped-row count, in the entry-condition sink format (merged by
        run_arm.aggregate_sink). Written only when the sink is set AND rule B is active."""
        sink = os.environ.get("FT_ENTRY_CONDITION_SINK")
        if not sink:
            return
        counts = {}
        if self.MOMENTUM_RECENCY_BARS > 0:
            counts["coexist_recency_refused"] = {
                "pass": int(getattr(self, "_st_coexist_refused", 0)), "total": int(rows)}
        if self.REVERSION_REGIME_GATE is not None:  # 026: gate liveness
            for k, v in (getattr(self, "_st_regime_counts", None) or {}).items():
                counts[k] = {"pass": int(v), "total": int(rows)}
        if not counts:
            return
        out = Path(sink)
        out.mkdir(parents=True, exist_ok=True)
        safe = pair.replace("/", "_").replace(":", "_")
        (out / f"{self.__class__.__name__}_{safe}_coexist.json").write_text(
            json.dumps({"pair": pair, "counts": counts})
        )


_apply_preset(SimpleTriggerStrategy)
validate_family_exits(SimpleTriggerStrategy)
