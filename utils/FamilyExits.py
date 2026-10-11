"""Per-trade exit parameters by entry family (feature 024, contract C2).

A trade is tagged at entry with `"<family>:<trigger>[+ov]"` (`enter_tag`; `+ov` marks a bar where
a trigger of another family also fired and priority decided the tag). Every exit and stop decision
in `BaseStrategy` reads its numbers from `_exit_params(trade)` (024 C1). This mixin overrides that
hook: it takes the strategy's own parameters and replaces the fields the trade's family overrides.

    class MyStrategy(FamilyExitMixin, ..., BaseStrategy):
        FAMILY_EXITS = {"momentum": {"take_profit": 10.0, "unclog_12h_hours": 24.0}}

STRATEGY-AGNOSTIC: no trigger names, no registry, no SimpleTrigger import -- it reads
`FAMILY_EXITS` and the tag only, so a later NNNC feature can adopt it unchanged (FR-015).

SHARED BY DEFINITION (refused in a family set, FR-002): the grace stop and ROI. Entry settings
are not `ExitParams` fields at all.

A trade with no recognised family uses the strategy's own (shared) parameters (FR-005). The
in-process `untagged_exits` counter is for tests and logs; the run record's `untagged_trades` is
computed from the backtest archive (contract C8), because this object lives in freqtrade's
process.
"""

from __future__ import annotations

import dataclasses
import warnings

from utils.ExitPolicy import ExitParams


FAMILIES = ("reversion", "momentum", "unassigned")

#: Shared across families by definition (FR-002).
FORBIDDEN_KEYS = frozenset({"stoploss_grace_hours", "stoploss_grace_level", "roi"})

#: Strategy-side override names -> the ExitParams field they configure (BaseStrategy._exit_params
#: mapping). Used only to refuse a key set both here and in a family set (F3).
_SELL_TO_FIELD = {
    "cexit_take_profit": "take_profit",
    "cexit_max_days": "max_hold_days",
    "cexit_enable_profit_checks": "cexit_enable_profit_checks",
}
_ATTR_TO_FIELD = {
    "atr_stoploss_multiplier": "atr_stoploss_multiplier",
    "atr_stoploss_cap": "atr_stoploss_cap",
    "atr_stoploss_floor": "atr_stoploss_floor",
    "use_atr_adaptive_stoploss": "use_atr_adaptive_stoploss",
    "use_volume_confirmation_stoploss": "use_volume_confirmation_stoploss",
    "stoploss": "static_stoploss",
}

_FIELDS = frozenset(f.name for f in dataclasses.fields(ExitParams))


class FamilyExitError(ValueError):
    """A family exit set that cannot be applied as declared. Raised at class creation."""


def family_of(tag) -> str | None:
    """The family prefix of `"<family>:<trigger>[+ov]"`, or None if absent or unknown."""
    if not tag or ":" not in tag:
        return None
    family = tag.split(":", 1)[0].strip()
    return family if family in FAMILIES else None


def _validate(cls) -> None:
    sets = getattr(cls, "FAMILY_EXITS", None) or {}
    unknown = sorted(set(sets) - set(FAMILIES))
    if unknown:
        raise FamilyExitError(f"{cls.__name__}: unknown family {unknown}; one of {FAMILIES}")
    if not any(sets.values()):
        return
    base = cls._exit_params(cls)  # class attributes + Parameter defaults; never instantiates
    explicit = {_SELL_TO_FIELD[k] for k in getattr(cls, "SELL_OVERRIDES", {}) if k in _SELL_TO_FIELD}
    explicit |= {_ATTR_TO_FIELD[k] for k in getattr(cls, "ATTR_OVERRIDES", {}) if k in _ATTR_TO_FIELD}
    for family, overrides in sets.items():
        for key in overrides:
            if key in FORBIDDEN_KEYS:
                raise FamilyExitError(
                    f"{cls.__name__}: {family}.{key} -- grace and ROI are shared by every "
                    f"family (024 FR-002); set them on the preset, not per family."
                )
            if key not in _FIELDS:
                raise FamilyExitError(f"{cls.__name__}: {family}.{key} is not an ExitParams field")
            if key in explicit:
                raise FamilyExitError(
                    f"{cls.__name__}: {key!r} is set both in FAMILY_EXITS[{family!r}] and via "
                    f"SELL_OVERRIDES/ATTR_OVERRIDES -- ambiguous precedence (024 F3). Override it "
                    f"per family in FAMILY_EXITS only."
                )
        try:
            params = dataclasses.replace(base, **overrides)
        except ValueError as exc:
            raise FamilyExitError(f"{cls.__name__}: {family} set is invalid: {exc}") from exc
        if params.unclog_12h_hours > params.max_hold_days * 24.0:
            raise FamilyExitError(
                f"{cls.__name__}: {family} unclog_12h_hours={params.unclog_12h_hours} exceeds the "
                f"max hold ({params.max_hold_days}d) -- it can never fire (INERT, 024 F5)."
            )
        if params.unclog_12h_hours > params.unclog_1d_hours:
            warnings.warn(
                f"{cls.__name__}: {family} unclog_12h_hours={params.unclog_12h_hours} > "
                f"unclog_1d_hours={params.unclog_1d_hours}: unclog_1d will usually fire first "
                f"(024 F5).",
                stacklevel=2,
            )


class FamilyExitMixin:
    """Choose `_exit_params` by the trade's entry family. Mix in AHEAD of the strategy base."""

    #: {family: {ExitParams field: value}}. Empty / all-empty = shared exits for every trade.
    FAMILY_EXITS: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _validate(cls)

    @property
    def untagged_exits(self) -> int:
        return getattr(self, "_untagged_exits", 0)

    def _exit_params(self, trade=None):
        base = super()._exit_params(trade)
        if trade is None:
            return base
        family = family_of(getattr(trade, "enter_tag", None))
        if family is None:
            self._untagged_exits = self.untagged_exits + 1
            return base
        overrides = (self.FAMILY_EXITS or {}).get(family) or {}
        return dataclasses.replace(base, **overrides) if overrides else base
