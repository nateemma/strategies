"""T015 — StopInvariantMixin (023 research R9, FR-017).

The mixin wraps a parent's `custom_stoploss` and halts the run on the defect classes this
repository has actually shipped: the post-grace sign flip (09db623) and a stop looser than its
documented bound. The bound is checked in PRICE space -- freqtrade's ratio is relative to
`current_rate`, so a correct entry-anchored stop on a winning trade has a ratio MORE negative
than the floor.
"""
from __future__ import annotations

import types
from datetime import datetime, timedelta, timezone

import pytest

from utils.ExitPolicy import STOP_AT_MARKET_RATIO
from utils.StopInvariants import StopInvariantError, StopInvariantMixin

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


class _Parent:
    VALUE = None

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit,
                        after_fill, **kwargs):
        return self.VALUE


class _Strat(StopInvariantMixin, _Parent):
    stoploss = -0.05
    use_atr_adaptive_stoploss = True
    atr_stoploss_floor = -0.04
    stoploss_grace_hours = 6.5
    stoploss_grace_level = -0.27


def _trade(open_rate=100.0, tid=1):
    return types.SimpleNamespace(id=tid, open_rate=open_rate, open_date_utc=T0)


def _call(s, value, current_rate, hours, after_fill=False, trade=None):
    s.VALUE = value
    return s.custom_stoploss("X/USDT", trade or _trade(), T0 + timedelta(hours=hours),
                             current_rate, 0.0, after_fill)


def test_a_positive_ratio_raises():
    # the 09db623 defect: a positive post-grace ratio that freqtrade's abs() turns looser
    with pytest.raises(StopInvariantError, match="sign"):
        _call(_Strat(), 0.06, 95.0, hours=7)


def test_no_change_sentinel_passes():
    s = _Strat()
    assert _call(s, 1.0, 100.0, hours=1) == 1.0
    assert _call(s, None, 100.0, hours=1) is None


def test_b_stop_price_below_zero_raises():
    with pytest.raises(StopInvariantError, match="price"):
        _call(_Strat(), -1.5, 100.0, hours=7)


def test_c_post_grace_bound_in_price_space():
    s = _Strat()
    # looser than open*(1+floor)=96: stop at 110*(1-0.20)=88 -> raise
    with pytest.raises(StopInvariantError, match="bound"):
        _call(s, -0.20, 110.0, hours=7, trade=_trade(tid=2))
    # MUST NOT raise: winning trade, correct entry-anchored stop at 96, ratio 96/110-1 = -0.1273
    ratio = 96.0 / 110.0 - 1.0
    assert ratio < s.atr_stoploss_floor            # the ratio LOOKS looser than the floor
    assert _call(s, ratio, 110.0, hours=7, trade=_trade(tid=3)) == ratio


def test_c_static_bound_when_not_adaptive():
    s = _Strat()
    s.use_atr_adaptive_stoploss = False
    _call(s, -0.05, 100.0, hours=7, trade=_trade(tid=4))           # exactly at bound: ok
    with pytest.raises(StopInvariantError, match="bound"):
        _call(s, -0.08, 100.0, hours=7, trade=_trade(tid=5))


def test_in_grace_wide_stop_is_allowed():
    s = _Strat()
    assert _call(s, -0.27, 100.0, hours=0, after_fill=True) == -0.27


def test_stop_at_market_clamp_is_exempt():
    s = _Strat()
    # trade already below its post-grace stop: the parent clamps to fire now
    assert _call(s, STOP_AT_MARKET_RATIO, 90.0, hours=7) == STOP_AT_MARKET_RATIO


def test_d_direction_across_grace_boundary():
    s = _Strat()
    t = _trade(tid=9)
    _call(s, -0.03, 100.0, hours=0, after_fill=True, trade=t)      # in-grace stop at 97
    # first post-grace stop at 105*(1-0.08)=96.6 < 97 while price 105 is above it -> raise
    with pytest.raises(StopInvariantError, match="direction"):
        _call(s, -0.08, 105.0, hours=7, trade=t)


def test_e_valid_value_returned_unchanged():
    s = _Strat()
    v = -0.035
    assert _call(s, v, 100.0, hours=7, trade=_trade(tid=11)) is v


def test_no_grace_configured_checks_from_the_start():
    s = _Strat()
    s.stoploss_grace_hours = types.SimpleNamespace(value=0.0)      # a Parameter-like object
    with pytest.raises(StopInvariantError, match="bound"):
        _call(s, -0.10, 100.0, hours=0, after_fill=True)


def test_c_adaptive_fallback_to_static_stoploss_is_within_bound():
    # adaptive with a TIGHT floor: _resolve_normal_stop falls back to `stoploss` (-0.05) when
    # ATR is unavailable, which is looser than the floor (-0.04) and still legitimate
    s = _Strat()
    assert _call(s, -0.05, 100.0, hours=7, trade=_trade(tid=21)) == -0.05
    with pytest.raises(StopInvariantError, match="bound"):
        _call(s, -0.051, 100.0, hours=7, trade=_trade(tid=22))
