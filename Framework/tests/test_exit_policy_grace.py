# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""T010 (specs/021-label-exit-coordination) — the stoploss-grace ASYMMETRY.

Two levels, both named explicitly because losing either would silently widen every
live stop with no test failing to say so (contracts/exit-policy.md: "losing that
asymmetry would widen every live stop silently"):

(a) `utils.ExitPolicy.effective_stop_fraction` -- the pure boundary: strictly BEFORE
    `stoploss_grace_hours` the wide grace level applies, AT and AFTER it the normal
    (ATR/static) stop applies. Tested at the exact boundary, not "close enough".

(b) `Framework.BaseStrategy.custom_stoploss` -- the live asymmetry the fixture in
    test_exit_policy_characterisation.py already encodes bar-by-bar but never names:
    inside the grace window the `after_fill=False` call returns `1.0` (freqtrade's
    "no change" sentinel) while the `after_fill=True` call returns the grace level
    itself. freqtrade only lets the INITIAL stop (set at `after_fill`) be arbitrarily
    wide; returning the grace level on the non-`after_fill` call would have no such
    license and is not exercised by the characterisation fixture's bar-for-bar
    equality check in the same explicit way.

Part (b) reuses the stub classes from test_exit_policy_characterisation.py by
importing them rather than duplicating them, per the task instruction -- importing
is NOT awkward here: the stubs are plain classes with no module-level side effects
(no file I/O happens until `_load`/`_cases` are called, which this test never does),
so the import is as cheap as defining them locally would have been.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.ExitPolicy import ExitParams, effective_stop_fraction  # noqa: E402
from Framework.BaseStrategy import BaseStrategy  # noqa: E402
from Framework.tests.test_exit_policy_characterisation import (  # noqa: E402
    _StubStrategy,
    _StubTrade,
)

import pandas as pd  # noqa: E402


# --------------------------------------------------------------------------------
# (a) utils.ExitPolicy.effective_stop_fraction -- the pure boundary
# --------------------------------------------------------------------------------

GRACE_PARAMS = ExitParams(
    stoploss_grace_hours=6.5,
    stoploss_grace_level=-0.27,
    use_atr_adaptive_stoploss=False,
    static_stoploss=-0.05,
)


def test_grace_applies_strictly_before_the_boundary():
    stop = effective_stop_fraction(
        age_hours=6.4999, atr_pct=0.0, rvol=None, params=GRACE_PARAMS
    )
    assert stop == GRACE_PARAMS.stoploss_grace_level


def test_normal_stop_applies_exactly_at_the_boundary():
    """`age_hours < stoploss_grace_hours` is the guard: AT 6.5 it is False, so the
    boundary bar already gets the normal (tightened) stop, not one extra bar of
    grace."""
    stop = effective_stop_fraction(
        age_hours=6.5, atr_pct=0.0, rvol=None, params=GRACE_PARAMS
    )
    assert stop == GRACE_PARAMS.static_stoploss


def test_normal_stop_applies_after_the_boundary():
    stop = effective_stop_fraction(
        age_hours=6.5001, atr_pct=0.0, rvol=None, params=GRACE_PARAMS
    )
    assert stop == GRACE_PARAMS.static_stoploss


def test_grace_disabled_when_grace_hours_is_zero():
    """`stoploss_grace_hours > 0.0` gates the whole branch -- age 0.0 would satisfy
    `age_hours < stoploss_grace_hours` if the gate were missing, so this is the case
    that would silently reappear if the `> 0.0` guard were ever dropped."""
    params = ExitParams(
        stoploss_grace_hours=0.0,
        stoploss_grace_level=-0.27,
        use_atr_adaptive_stoploss=False,
        static_stoploss=-0.05,
    )
    stop = effective_stop_fraction(age_hours=0.0, atr_pct=0.0, rvol=None, params=params)
    assert stop == params.static_stoploss


# --------------------------------------------------------------------------------
# (b) Framework.BaseStrategy.custom_stoploss -- the live asymmetry
# --------------------------------------------------------------------------------

_GRACE_OVERRIDES = {"stoploss_grace_hours": 6.5, "stoploss_grace_level": -0.27}


def _inside_grace_window_stop(after_fill: bool) -> float:
    """One bar of one pair, just enough for `custom_stoploss` to read -- same
    construction `test_exit_policy_characterisation.py` uses, imported rather than
    reconstructed by hand."""
    df = pd.DataFrame(
        {
            "close": [100.0] * 60,
            "volume": [1.0] * 60,
            "atr_pct_roll": [0.01] * 60,
        }
    )
    strat = _StubStrategy(df, **_GRACE_OVERRIDES)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trade = _StubTrade(now - timedelta(hours=3.0), 100.0)  # inside the 6.5h window
    return float(
        BaseStrategy.custom_stoploss(
            strat,
            pair="SOL/USDT",
            trade=trade,
            current_time=now,
            current_rate=100.0,
            current_profit=0.0,
            after_fill=after_fill,
        )
    )


def test_inside_grace_non_after_fill_returns_freqtrade_no_change_sentinel():
    assert _inside_grace_window_stop(after_fill=False) == 1.0


def test_inside_grace_after_fill_returns_the_grace_level():
    assert _inside_grace_window_stop(after_fill=True) == _GRACE_OVERRIDES["stoploss_grace_level"]


def test_grace_asymmetry_is_a_real_asymmetry():
    """Named explicitly, not just implied by the two tests above: the two calls
    inside the SAME window on the SAME bar must disagree. Losing this (e.g. by
    making the non-after_fill branch also return the grace level) would widen
    every live stop silently, per contracts/exit-policy.md."""
    assert _inside_grace_window_stop(after_fill=False) != _inside_grace_window_stop(
        after_fill=True
    )
