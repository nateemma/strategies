"""The shared entry framework of NNNC and SimpleTrigger (feature 029, contract I1).

One strategy-agnostic layer between a strategy's ENTRY SOURCES and `BaseStrategy`'s unchanged
guard chain, stop and exits. It was `SimpleTriggerStrategy`'s own code (023-026); NNNC now derives
from it too, so the two families cannot drift apart (023 FR-007a).

ENTRY SOURCES (`ENTRY_MODE`)
  "triggers"           the OR of the enabled registry triggers (`research.outcomes.triggers`),
                       tagged `<trigger family>:<trigger>[+ov]` -- SimpleTrigger.
  "model"              the next class's `get_entry_conditions` (BaseNNStrategy: the model's
                       prediction against its threshold), returned UNCHANGED. Tagged
                       `<label family>:model` when the training label implies a family
                       (`utils.LabelFamilies`, decision A′), else BaseStrategy's "model_entry ".
                       NNNC's default: exactly today's entry.
  "trigger_and_model"  model AND OR(triggers); tagged as "model". A declared switch, off by
                       default (029 FR-003).

FRAMEWORK FEATURES, each off by default and declared on the class:
  per-family exits    FAMILY_EXIT_OVERRIDES, merged PER FAMILY, PER KEY along the MRO (025 G1)
                      into FAMILY_EXITS, which `FamilyExitMixin` applies per trade by entry tag;
  rule A              FAMILY_PAIR_CAP (`FamilyCoexistenceMixin`);
  rule B              MOMENTUM_RECENCY_BARS (025) -- trigger mode only;
  pre-emption         MOMENTUM_PREEMPT (026 addendum 2) -- trigger mode only;
  regime gate         REVERSION_REGIME_GATE (026): reversion-tagged entries dropped (or, in
                      trigger mode, handed to a momentum trigger on the same bar) in the "off"
                      state; needs a >= 4096-bar startup;
  stop invariants     `StopInvariantMixin` on every strategy (023 FR-017);
  held-pair add       HELD_PAIR_ADD (031): a fresh entry signal on a pair that holds an open long
                      trade adds one stake to it (freqtrade position adjustment) instead of being
                      ignored; trade clocks then restart at the last fill
                      (`BaseStrategy.trade_clock_start`).

Validation happens at class creation. A strategy that resolves its own attributes first (a
SimpleTrigger preset) sets `_ENTRY_FRAMEWORK_AUTO = False` and calls `_finalise_entry_framework`
itself. Errors are raised as `ENTRY_ERROR` (SimpleTrigger keeps its `PresetError`).

Strategy-agnostic: names families and entry sources, never a strategy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.FamilyCoexistence import FamilyCoexistenceMixin
from utils.FamilyExits import FAMILIES, FamilyExitMixin, family_of, validate_family_exits
from utils.StopInvariants import StopInvariantMixin


ENTRY_MODES = ("triggers", "model", "trigger_and_model")
#: 031: classes on which `_resolve_held_pair_add` set freqtrade's position-adjust attributes.
_HP_SET: set = set()
#: MODEL_ENTRY_FAMILY sentinel: resolve from the class's training label.
AUTO = "auto"
#: 026: a market-wide regime instrument needs ~2,900 bars of history to be defined from a fold's
#: first bar (research R4/H1); a power of 2 per the BaseStrategy startup comment.
REGIME_GATE_MIN_STARTUP = 4096


class EntryFrameworkError(ValueError):
    """An entry-framework declaration that cannot be applied as declared. Raised at class
    creation."""


class EntryFrameworkMixin(FamilyCoexistenceMixin, FamilyExitMixin, StopInvariantMixin):
    ENTRY_MODE: str = "triggers"
    #: Error type for declaration problems (SimpleTrigger sets its PresetError).
    ENTRY_ERROR: type = EntryFrameworkError

    #: Trigger source. Exactly one of TRIGGER_NAME / TRIGGERS when triggers are used.
    TRIGGER_NAME: str | None = None
    TRIGGERS: tuple | None = None
    #: Triggers not evaluated at all -- disabling one side reproduces the other exactly (024).
    DISABLED_TRIGGERS: tuple = ()
    #: On a bar where triggers of several families fire, the earliest family here wins the tag.
    FAMILY_PRIORITY: tuple = ("reversion", "momentum", "unassigned")

    #: 025 rule B: drop a reversion-only entry if a momentum-family trigger fired on the pair
    #: within the last N bars (current included); while active, overlap bars go to momentum.
    #: 0 = off. Rule A (FAMILY_PAIR_CAP) comes from FamilyCoexistenceMixin.
    MOMENTUM_RECENCY_BARS: int = 0
    #: 026 addendum 2: momentum never waits for a pair reversion holds. A fillable momentum signal
    #: on a pair held by a reversion trade exits that trade at the next open ("momentum_preempt");
    #: momentum enters on the following bar if its signal is still live (freqtrade evaluates a
    #: bar's entry before its exits). No late copy of the signal: an extra entry signal suppresses
    #: ROI on any open trade (ignore_roi_if_entry_signal) and broke the null check. Overlap bars
    #: go to momentum. False = off.
    MOMENTUM_PREEMPT: bool = False
    #: 026: None, or {"instrument": <utils.RegimeState id>, "threshold": float,
    #: "off_when": "ge"|"lt"}. In the "off" state reversion-tagged entries are dropped (trigger
    #: mode: overlap entries go to momentum); NaN state = "on". Requires a >= 4096-bar startup.
    REVERSION_REGIME_GATE: dict | None = None

    #: {family: {ExitParams field: value}}, merged along the MRO into FAMILY_EXITS (024/025).
    FAMILY_EXIT_OVERRIDES: dict = {}
    FAMILY_EXITS: dict = {}
    #: The family a MODEL entry is tagged with: AUTO (from TRAINING_TYPE via
    #: utils.LabelFamilies), a family name, or None (untagged, on the shared exit).
    MODEL_ENTRY_FAMILY = AUTO

    #: 031: None = off. A dict (even {}) = on, merged PER KEY along the MRO over
    #: HELD_PAIR_ADD_DEFAULTS into HELD_PAIR_ADD_RESOLVED; a subclass's None turns it off.
    HELD_PAIR_ADD: dict | None = None
    HELD_PAIR_ADD_DEFAULTS = {"max_adds": 1, "min_age_hours": 1.0, "under_water_only": False}
    #: 031 follow-up keys: resolved (and so fingerprinted) only when set to a non-default value,
    #: so the original 031 arms keep their fingerprints. `in_profit_only`: add only to a held trade
    #: in profit (pyramiding). `add_fraction`: the add is this fraction of the fresh-entry stake.
    HELD_PAIR_ADD_OPTIONAL = {"in_profit_only": False, "add_fraction": 1.0}
    HELD_PAIR_ADD_RESOLVED: dict | None = None

    # FAMILY_EXITS is resolved here, so FamilyExitMixin must not validate before it is.
    _FAMILY_VALIDATE_ON_SUBCLASS = False
    #: False when a subclass hierarchy resolves its attributes first (SimpleTrigger presets).
    _ENTRY_FRAMEWORK_AUTO = True

    # Feature 030: the model key (Framework/ModelKey.py). This class declares the names it
    # owns; the lists merge along the MRO. ATTRIBUTES/METHODS shape trained weights;
    # IGNORED act only after the model. The audit test keeps every name classified.
    MODEL_KEY_IGNORED = (
        "ENTRY_MODE", "FAMILY_EXIT_OVERRIDES", "FAMILY_EXITS", "MODEL_ENTRY_FAMILY",
        "REVERSION_REGIME_GATE", "TRIGGER_NAME", "TRIGGERS", "DISABLED_TRIGGERS",
        "FAMILY_PRIORITY", "MOMENTUM_PREEMPT", "MOMENTUM_RECENCY_BARS", "_ENTRY_FRAMEWORK_AUTO",
        "_FAMILY_VALIDATE_ON_SUBCLASS", "HELD_PAIR_ADD", "HELD_PAIR_ADD_DEFAULTS",
        "HELD_PAIR_ADD_RESOLVED", "HELD_PAIR_ADD_OPTIONAL", "position_adjustment_enable",
        "max_entry_position_adjustment", "TRADE_CLOCK_FROM_LAST_FILL",
    )
    MODEL_KEY_METHODS_IGNORED = ("adjust_trade_position", "_resolve_held_pair_add")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._ENTRY_FRAMEWORK_AUTO:
            cls._finalise_entry_framework({})

    # ----------------------------------------------------------------------- class creation

    @classmethod
    def _finalise_entry_framework(cls, base_family_exits: dict) -> None:
        """Resolve FAMILY_EXITS over `base_family_exits`, validate every declaration."""
        cls.FAMILY_EXITS = cls._resolve_family_exits(base_family_exits)
        cls._validate_entry_framework()
        validate_family_exits(cls)
        cls._resolve_held_pair_add()

    @classmethod
    def _resolve_held_pair_add(cls) -> None:
        """031: merge HELD_PAIR_ADD along the MRO, validate it, and switch freqtrade's position
        adjustment on for the class when it is on. Off touches no freqtrade attribute."""
        merged = None
        for klass in reversed(cls.__mro__):
            if "HELD_PAIR_ADD" not in klass.__dict__:
                continue
            decl = klass.__dict__["HELD_PAIR_ADD"]
            if decl is None:
                merged = None
                continue
            if not isinstance(decl, dict):
                raise cls.ENTRY_ERROR(
                    f"{cls.__name__}: HELD_PAIR_ADD must be None or a dict, got {decl!r}")
            merged = {**(merged or {}), **decl}
        if merged is None:
            cls._held_pair_add_off()
            return
        unknown = sorted(set(merged) - set(cls.HELD_PAIR_ADD_DEFAULTS)
                         - set(cls.HELD_PAIR_ADD_OPTIONAL))
        if unknown:
            raise cls.ENTRY_ERROR(f"{cls.__name__}: HELD_PAIR_ADD unknown keys {unknown}")
        r = {**cls.HELD_PAIR_ADD_DEFAULTS,
             **{k: v for k, v in merged.items() if k in cls.HELD_PAIR_ADD_DEFAULTS}}
        if type(r["max_adds"]) is not int or r["max_adds"] < 1:
            raise cls.ENTRY_ERROR(
                f"{cls.__name__}: HELD_PAIR_ADD max_adds must be an int >= 1, got {r['max_adds']!r}")
        if type(r["min_age_hours"]) not in (int, float) or r["min_age_hours"] < 0:
            raise cls.ENTRY_ERROR(
                f"{cls.__name__}: HELD_PAIR_ADD min_age_hours must be a number >= 0, got "
                f"{r['min_age_hours']!r}")
        if type(r["under_water_only"]) is not bool:
            raise cls.ENTRY_ERROR(
                f"{cls.__name__}: HELD_PAIR_ADD under_water_only must be a bool, got "
                f"{r['under_water_only']!r}")
        r["min_age_hours"] = float(r["min_age_hours"])
        for k, default in cls.HELD_PAIR_ADD_OPTIONAL.items():
            if k in merged and merged[k] != default:
                r[k] = merged[k]
        if type(r.get("in_profit_only", False)) is not bool:
            raise cls.ENTRY_ERROR(
                f"{cls.__name__}: HELD_PAIR_ADD in_profit_only must be a bool, got "
                f"{r['in_profit_only']!r}")
        if r.get("in_profit_only") and r["under_water_only"]:
            raise cls.ENTRY_ERROR(
                f"{cls.__name__}: HELD_PAIR_ADD in_profit_only and under_water_only exclude "
                f"each other")
        if "add_fraction" in r:
            f = r["add_fraction"]
            if type(f) not in (int, float) or not 0 < f <= 1:
                raise cls.ENTRY_ERROR(
                    f"{cls.__name__}: HELD_PAIR_ADD add_fraction must be in (0, 1], got {f!r}")
            r["add_fraction"] = float(f)
        cls.HELD_PAIR_ADD_RESOLVED = r
        cls.position_adjustment_enable = True
        cls.max_entry_position_adjustment = r["max_adds"]
        cls.TRADE_CLOCK_FROM_LAST_FILL = True
        _HP_SET.add(cls)

    @classmethod
    def _held_pair_add_off(cls) -> None:
        """Off. A class whose ancestor turned the switch on gets back the freqtrade attributes it
        would have had without it -- from the nearest class that set them itself (e.g. another
        position-adjust mixin), else freqtrade's defaults -- and its clocks count from the open."""
        cls.HELD_PAIR_ADD_RESOLVED = None
        if not any(k in _HP_SET for k in cls.__mro__[1:]):
            return
        for name in ("position_adjustment_enable", "max_entry_position_adjustment"):
            owner = next(k for k in cls.__mro__[1:] if name in k.__dict__ and k not in _HP_SET)
            setattr(cls, name, owner.__dict__[name])
        cls.TRADE_CLOCK_FROM_LAST_FILL = False
        _HP_SET.discard(cls)

    @classmethod
    def _resolve_family_exits(cls, base: dict) -> dict:
        """025 G1: per-family sets = `base`, merged PER FAMILY, PER KEY with every class's
        FAMILY_EXIT_OVERRIDES, ancestors first -- a subclass adds to its parent's overrides."""
        families = {f: dict(v) for f, v in base.items()}
        merged: dict = {}
        for klass in reversed(cls.__mro__):
            for family, overrides in (klass.__dict__.get("FAMILY_EXIT_OVERRIDES") or {}).items():
                merged.setdefault(family, {}).update(overrides)
        unknown = sorted(set(merged) - set(FAMILIES))
        if unknown:
            raise cls.ENTRY_ERROR(
                f"{cls.__name__}: FAMILY_EXIT_OVERRIDES unknown family {unknown}"
            )
        for family, overrides in merged.items():
            families[family] = {**families.get(family, {}), **overrides}
        return families

    @classmethod
    def _trigger_names(cls) -> tuple:
        return tuple(cls.TRIGGERS or ((cls.TRIGGER_NAME,) if cls.TRIGGER_NAME else ()))

    @classmethod
    def _model_family(cls) -> str | None:
        fam = cls.MODEL_ENTRY_FAMILY
        if fam == AUTO:
            from utils.LabelFamilies import label_family

            return label_family(getattr(cls, "TRAINING_TYPE", None))
        return fam

    @classmethod
    def _validate_entry_framework(cls) -> None:
        err = cls.ENTRY_ERROR
        mode = cls.ENTRY_MODE
        if mode not in ENTRY_MODES:
            raise err(f"{cls.__name__}: ENTRY_MODE {mode!r} not in {ENTRY_MODES}")
        if cls.TRIGGER_NAME and cls.TRIGGERS:
            raise err(f"{cls.__name__}: set TRIGGER_NAME or TRIGGERS, not both")
        names = cls._trigger_names()
        if mode == "model" and names:
            raise err(
                f"{cls.__name__}: ENTRY_MODE 'model' with a trigger declared is ambiguous; use "
                f"'trigger_and_model' to require the trigger"
            )
        if mode == "trigger_and_model" and not names:
            raise err(f"{cls.__name__}: ENTRY_MODE 'trigger_and_model' needs a trigger")
        fam = cls.MODEL_ENTRY_FAMILY
        if fam not in (AUTO, None) and fam not in FAMILIES:
            raise err(f"{cls.__name__}: MODEL_ENTRY_FAMILY {fam!r} not in {FAMILIES}")
        stray = sorted(set(cls.DISABLED_TRIGGERS) - set(names))
        if stray:
            raise err(f"{cls.__name__}: DISABLED_TRIGGERS {stray} not in TRIGGERS")
        if names and set(names) <= set(cls.DISABLED_TRIGGERS):
            raise err(f"{cls.__name__}: DISABLED_TRIGGERS disables every trigger")
        if sorted(cls.FAMILY_PRIORITY) != sorted(FAMILIES):
            raise err(
                f"{cls.__name__}: FAMILY_PRIORITY {cls.FAMILY_PRIORITY} is not a permutation of "
                f"{FAMILIES}"
            )
        cls._validate_regime_gate(names)
        window = cls.MOMENTUM_RECENCY_BARS
        if type(window) is not int or window < 0:
            raise err(
                f"{cls.__name__}: MOMENTUM_RECENCY_BARS must be an int >= 0, got {window!r}"
            )
        if window > 0 and cls.MOMENTUM_PREEMPT:
            raise err(
                f"{cls.__name__}: MOMENTUM_RECENCY_BARS and MOMENTUM_PREEMPT are alternative "
                f"momentum-precedence rules; set one"
            )
        for rule, on in (("MOMENTUM_RECENCY_BARS", window > 0),
                         ("MOMENTUM_PREEMPT", bool(cls.MOMENTUM_PREEMPT))):
            if not on:
                continue
            if mode != "triggers":
                raise err(
                    f"{cls.__name__}: {rule} arbitrates between TRIGGER families; it needs "
                    f"ENTRY_MODE 'triggers'"
                )
            if not any(f == "momentum" for f in cls._enabled_trigger_families(names)):
                raise err(
                    f"{cls.__name__}: {rule} needs an enabled momentum "
                    f"trigger; without one the rule is INERT by construction"
                )

    @classmethod
    def _enabled_trigger_families(cls, names) -> list:
        from research.outcomes import triggers as _triggers

        _triggers.ensure_builtins()
        return [_triggers.get(t).family for t in names if t not in cls.DISABLED_TRIGGERS]

    @classmethod
    def _validate_regime_gate(cls, names) -> None:
        gate = cls.REVERSION_REGIME_GATE
        if gate is None:
            return
        err = cls.ENTRY_ERROR
        from utils import RegimeState as _rs

        if not isinstance(gate, dict) or gate.get("instrument") not in _rs.INSTRUMENTS:
            raise err(
                f"{cls.__name__}: REVERSION_REGIME_GATE instrument must be one of "
                f"{sorted(_rs.INSTRUMENTS)}, got {gate!r}"
            )
        if gate.get("off_when") not in ("ge", "lt"):
            raise err(f"{cls.__name__}: REVERSION_REGIME_GATE off_when must be ge/lt")
        thr = gate.get("threshold")
        if not isinstance(thr, (int, float)) or not np.isfinite(thr):
            raise err(f"{cls.__name__}: REVERSION_REGIME_GATE threshold must be finite")
        if int(cls.startup_candle_count) < REGIME_GATE_MIN_STARTUP:
            raise err(
                f"{cls.__name__}: REVERSION_REGIME_GATE needs startup_candle_count >= "
                f"{REGIME_GATE_MIN_STARTUP} (got {cls.startup_candle_count}); a shorter warm-up "
                f"leaves the instrument undefined for part of each fold (026 H1)"
            )
        if cls.ENTRY_MODE == "triggers":
            reversion = any(f == "reversion" for f in cls._enabled_trigger_families(names))
            what = "an enabled reversion trigger"
        else:
            reversion = cls._model_family() == "reversion"
            what = "model entries tagged reversion (MODEL_ENTRY_FAMILY)"
        if not reversion:
            raise err(
                f"{cls.__name__}: REVERSION_REGIME_GATE needs {what}; without one it is INERT "
                f"by construction"
            )

    # ---------------------------------------------------------------------------- entries

    def _refuse(self, reason: str, n: int) -> np.ndarray:
        if not getattr(self, "_st_refusal_reported", False):
            self._st_refusal_reported = True
            print(
                f"*** {self.__class__.__name__} enters NO trades -- {reason}",
                flush=True,
            )
        return np.zeros(n, dtype=bool)

    def _enabled_triggers(self) -> tuple:
        return tuple(n for n in self._trigger_names() if n not in self.DISABLED_TRIGGERS)

    def _evaluate_triggers(self, dataframe):
        """(names, masks, families) or a refusal reason string."""
        names = self._enabled_triggers()
        if not names:
            return ("TRIGGER_NAME is not set; a trigger-less arm would silently "
                    "measure the guards instead")
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
                return f"trigger {name!r} is not runnable: {why}"
        try:
            from research.outcomes import triggers as _triggers

            _triggers.ensure_builtins()
            masks = [_triggers.evaluate(name, dataframe).to_numpy() for name in names]
            families = [_triggers.get(name).family for name in names]
        except Exception as exc:
            return (f"could not evaluate trigger(s) {names} ({type(exc).__name__}: "
                    f"{exc}); no fallback definition is used")
        return names, masks, families

    def get_entry_conditions(self, dataframe):
        n = len(dataframe)
        mode = self.ENTRY_MODE
        if mode == "model":
            mask = super().get_entry_conditions(dataframe)
            self._st_tags = self._model_tags(mask, n)
            if self.REVERSION_REGIME_GATE is not None and mask is not None:
                self._st_tags, mask = self._apply_regime_gate(
                    dataframe, (), [], [], self._st_tags, mask)
            return mask
        got = self._evaluate_triggers(dataframe)
        if isinstance(got, str):
            return self._refuse(got, n)
        names, masks, families = got
        if mode == "trigger_and_model":
            model = super().get_entry_conditions(dataframe)
            model_b = (np.zeros(n, dtype=bool) if model is None
                       else np.asarray(model, dtype=bool))
            trig = np.logical_or.reduce(masks)
            mask = model_b & trig
            self._st_trigger_gate_refused = int((model_b & ~trig).sum())
            tags = self._model_tags(mask, n)
            if self.REVERSION_REGIME_GATE is not None:
                tags, mask = self._apply_regime_gate(dataframe, (), [], [], tags, mask)
            self._st_tags = tags
            return mask
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

    def _model_tags(self, mask, n: int):
        """`<label family>:model` on model-entry rows, or None (keep "model_entry ")."""
        fam = self._model_family()
        if not fam or mask is None:
            return None
        tags = np.full(n, "", dtype=object)
        tags[np.asarray(mask, dtype=bool)] = f"{fam}:model"
        return tags

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

    def populate_entry_trend(self, dataframe, metadata: dict):
        """BaseStrategy's entry, then the family tag on every signal row (024 C4)."""
        self._st_tags = None
        dataframe = super().populate_entry_trend(dataframe, metadata)
        tags = self._st_tags
        if tags is not None and len(tags) == len(dataframe):
            signal = tags != ""
            dataframe.loc[signal, "enter_tag"] = tags[signal]
        self._sink_coexistence(metadata.get("pair", "?"), len(dataframe))
        return dataframe

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
            from utils.RegimeState import REGIME_CONTEXT_PAIR

            return [(REGIME_CONTEXT_PAIR, self.timeframe)]
        return super().informative_pairs()

    def _regime_state(self) -> pd.Series:
        """The gate's instrument over the whitelist (and BTC), cached per latest bar."""
        from utils import RegimeState as _rs

        gate = self.REVERSION_REGIME_GATE
        frames = {p: self.dp.get_pair_dataframe(p, self.timeframe)
                  for p in self.dp.current_whitelist()}
        btc = (self.dp.get_pair_dataframe(_rs.REGIME_CONTEXT_PAIR, self.timeframe)
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
        if tags is None:
            tags = np.full(n, "", dtype=object)
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

    # --------------------------------------------------------------------- 031 held-pair add

    #: AddDecision refusal reasons, in check order (data-model.md).
    HP_REASONS = ("not_long", "open_order", "max_adds", "row_misaligned", "no_signal",
                  "too_young", "in_profit", "not_in_profit", "exit_due", "thin", "below_min")

    def _hp_count(self, key: str) -> None:
        c = self.__dict__.get("_hp_counts")
        if c is None:
            c = self._hp_counts = {}
            import atexit

            atexit.register(self._hp_sink)
        c[key] = c.get(key, 0) + 1

    def _hp_refuse(self, reason: str):
        self._hp_count(f"refused_{reason}")
        return None

    def _hp_sink(self) -> None:
        """Write the add counters into the run's entry-condition sink (`run_arm` merges every
        `*.json` there into `entry_condition_counts`; keys are prefixed `hp_`). The switch must
        be on and the sink set."""
        sink = os.environ.get("FT_ENTRY_CONDITION_SINK")
        counts = self.__dict__.get("_hp_counts") or {}
        if not sink or not counts or self.HELD_PAIR_ADD_RESOLVED is None:
            return
        calls = int(sum(counts.values()))
        out = Path(sink)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{self.__class__.__name__}_hp_add.json").write_text(json.dumps({
            "counts": {f"hp_{k}": {"pass": int(v), "total": calls}
                       for k, v in sorted(counts.items())}}))

    def _hp_exit_due(self, trade, current_time, current_rate, current_profit) -> bool:
        """Would the held trade exit at this candle's open without the add? The strategy's own
        custom exit, or ROI when an entry signal does not suppress it."""
        if self.custom_exit(trade.pair, trade, current_time, current_rate, current_profit):
            return True
        if not getattr(self, "ignore_roi_if_entry_signal", False):
            return bool(self.min_roi_reached(trade, current_profit, current_time))
        return False

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit,
                              min_stake, max_stake, current_entry_rate=None,
                              current_exit_rate=None, current_entry_profit=None,
                              current_exit_profit=None, **kwargs):
        """031: add one stake to a held trade when the strategy's own entry signal fires on its
        pair (data-model AddDecision; checks in order, the first failure refuses).

        freqtrade calls this at the open of each candle, before the exit checks, with
        `current_rate` = that open and `max_stake` already bounded by the available wallet. The
        signal candle is the one before (`current_time - timeframe`); a fresh entry would fill
        at this same open. freqtrade does not call `custom_stake_amount` or
        `confirm_trade_entry` for an add, so the liquidity cap and the quote-volume floor are
        applied here.
        """
        hp = self.HELD_PAIR_ADD_RESOLVED
        if hp is None:
            return None  # off: never counted, never sinks
        if getattr(trade, "is_short", False):
            return self._hp_refuse("not_long")
        if getattr(trade, "has_open_orders", False):
            return self._hp_refuse("open_order")
        if trade.nr_of_successful_entries - 1 >= hp["max_adds"]:
            return self._hp_refuse("max_adds")
        from datetime import timedelta

        from freqtrade.exchange import timeframe_to_minutes

        df, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
        signal = current_time - timedelta(minutes=timeframe_to_minutes(self.timeframe))
        if df is None or len(df) == 0 or df["date"].iloc[-1] != signal:
            return self._hp_refuse("row_misaligned")
        last = df.iloc[-1]
        if last.get("enter_long") != 1:
            return self._hp_refuse("no_signal")
        age_h = (current_time - trade.date_last_filled_utc).total_seconds() / 3600.0
        if age_h <= 0.0 or age_h < hp["min_age_hours"]:
            return self._hp_refuse("too_young")
        if hp["under_water_only"] and not current_profit < 0:
            return self._hp_refuse("in_profit")
        if hp.get("in_profit_only") and not current_profit > 0:
            return self._hp_refuse("not_in_profit")
        # The exit wins (031 follow-up fidelity diagnosis): freqtrade checks the add BEFORE this
        # candle's exits, so an add on a candle where the held trade's own exit fires at this open
        # would average the price, restart the clock -- and the 1 h minimum hold would then veto
        # the discretionary exit (take-profit), keeping the trade open. Refuse the add instead.
        if self._hp_exit_due(trade, current_time, current_rate, current_profit):
            return self._hp_refuse("exit_due")
        quote_volume = float(last["volume"]) * float(last["close"])
        if quote_volume < self.MIN_QUOTE_VOLUME:
            return self._hp_refuse("thin")
        try:
            proposed = self.wallets.get_trade_stake_amount(
                trade.pair, self.config["max_open_trades"], update=False)
        except Exception:  # noqa: BLE001 -- no wallet (tests) or freqtrade refuses a proposal
            proposed = float(self.config["stake_amount"])
        stake = min(float(proposed) * hp.get("add_fraction", 1.0),
                    quote_volume / self.QUOTE_VOLUME_HEADROOM_MULT, float(max_stake))
        if stake <= 0 or (min_stake and stake < min_stake):
            return self._hp_refuse("below_min")
        self._hp_count("adds")
        return stake, "hp_add"

    # ------------------------------------------------------------------------- liveness sink

    def _sink_coexistence(self, pair: str, rows: int) -> None:
        """025 C3 / 026 / 029: per-rule dropped-row counts, in the entry-condition sink format
        (merged by run_arm.aggregate_sink). Written only when the sink is set AND a rule is
        active."""
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
        if self.ENTRY_MODE == "trigger_and_model":  # 029: trigger-gate liveness
            counts["trigger_gate_refused"] = {
                "pass": int(getattr(self, "_st_trigger_gate_refused", 0)), "total": int(rows)}
        if not counts:
            return
        out = Path(sink)
        out.mkdir(parents=True, exist_ok=True)
        safe = pair.replace("/", "_").replace(":", "_")
        (out / f"{self.__class__.__name__}_{safe}_coexist.json").write_text(
            json.dumps({"pair": pair, "counts": counts})
        )
