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

import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame


sys.path.append(str(Path(__file__).parent.parent))

from Framework.BaseStrategy import BaseStrategy, StrategyConfig
from SimpleTrigger import presets as _presets
from utils.FamilyExits import FamilyExitMixin, validate_family_exits
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
    families = _presets.family_exits(preset)
    unknown = sorted(set(cls.FAMILY_EXIT_OVERRIDES) - set(families))
    if unknown:
        raise _presets.PresetError(
            f"{cls.__name__}: FAMILY_EXIT_OVERRIDES unknown family {unknown}"
        )
    for family, overrides in cls.FAMILY_EXIT_OVERRIDES.items():
        families[family] = {**families[family], **overrides}
    cls.FAMILY_EXITS = families
    _validate_triggers(cls)


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


class SimpleTriggerStrategy(FamilyExitMixin, StopInvariantMixin, BaseStrategy):
    """Trigger + framework preset, no model. Subclass and set TRIGGER_NAME."""

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
        self._st_tags = self._tags(names, masks, families, n)
        return np.logical_or.reduce(masks)

    def _tags(self, names, masks, families, n) -> np.ndarray:
        """`<family>:<trigger>` per signal row, by FAMILY_PRIORITY then trigger order; `+ov` when
        a trigger of ANOTHER family also fired on that row (024 C4, F4)."""
        rank = {f: i for i, f in enumerate(self.FAMILY_PRIORITY)}
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
        return dataframe


_apply_preset(SimpleTriggerStrategy)
validate_family_exits(SimpleTriggerStrategy)
