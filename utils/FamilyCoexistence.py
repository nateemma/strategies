"""Coexistence between entry families: a per-family pair cap (feature 025, rule A, contract C1).

freqtrade allows one open trade per pair, and in 024's combined arm the slot cap never bound: every
lost entry was a pair already held -- mostly by the OTHER family. Capping how many pairs one family
may hold at once keeps at least `n_pairs - cap` pairs free of it for the other family.

    class MyStrategy(FamilyCoexistenceMixin, ..., BaseStrategy):
        FAMILY_PAIR_CAP = 5

The cap refuses only NEW entries (it never closes or evicts a trade), and only after the parent's
own checks pass -- the volume-headroom check in `BaseStrategy.confirm_trade_entry` runs first and
unchanged. It reads entry tags only (`"<family>:<trigger>[+ov]"`, feature 024), never pair identity
(constitution X), and imports nothing strategy-specific, so a later NNNC feature can adopt it.

Refusals are appended to `$FT_ENTRY_CONDITION_SINK/coexist_events.jsonl` when that sink is set
(feature 023 C4 path); `regime/run_arm.py` aggregates them into the record's
`coexistence_refusals`. Counters on this object would never leave freqtrade's process (024 F2).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from utils.FamilyExits import family_of


class FamilyCoexistenceError(ValueError):
    """A coexistence setting that cannot be applied as declared. Raised at class creation."""


def _open_trades():
    """Open trades, backtest included (freqtrade's documented accessor). Patched in tests."""
    from freqtrade.persistence import Trade

    return Trade.get_trades_proxy(is_open=True)


def _log_refusal(rule: str, family: str) -> None:
    sink = os.environ.get("FT_ENTRY_CONDITION_SINK")
    if not sink:
        return
    with (Path(sink) / "coexist_events.jsonl").open("a") as fh:
        fh.write(json.dumps({"rule": rule, "family": family}) + "\n")


class FamilyCoexistenceMixin:
    """Refuse a family's new entry while it already holds `FAMILY_PAIR_CAP` pairs."""

    #: None = no cap (neutral). An int in 1..9: max open pairs per family, same for every family.
    FAMILY_PAIR_CAP: int | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cap = cls.FAMILY_PAIR_CAP
        if cap is not None and (type(cap) is not int or not 1 <= cap <= 9):
            raise FamilyCoexistenceError(
                f"{cls.__name__}: FAMILY_PAIR_CAP must be None or an int in 1..9, got {cap!r}"
            )

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time,
                            entry_tag, side, **kwargs) -> bool:
        ok = super().confirm_trade_entry(
            pair, order_type, amount, rate, time_in_force, current_time, entry_tag, side, **kwargs
        )
        cap = self.FAMILY_PAIR_CAP
        if not ok or cap is None:
            return ok
        family = family_of(entry_tag)
        if family is None:
            return ok
        held = sum(1 for t in _open_trades() if family_of(getattr(t, "enter_tag", None)) == family)
        if held >= cap:
            _log_refusal("pair_cap", family)
            return False
        return ok
