"""027 T002: path metrics and replays on synthetic 15m paths (research R2/R4)."""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from research.exit_coupling import paths as P
from utils.ExitPolicy import ExitParams, net_from_price_ratio


BASE = ExitParams(
    atr_stoploss_multiplier=4.0, atr_stoploss_cap=-0.04, atr_stoploss_floor=-0.06,
    use_atr_adaptive_stoploss=True, use_volume_confirmation_stoploss=False,
    stoploss_grace_hours=6.5, stoploss_grace_level=-0.27, take_profit=10.0,
    unclog_12h_hours=1e9, unclog_1d_hours=1e9, max_hold_days=10,
)
GRACE_BARS = 26  # 6.5 h


def _bars(n=1200, price=100.0, atr=0.005):
    o = np.full(n, price)
    return P.Bars(open=o.copy(), high=o * 1.001, low=o * 0.999, close=o.copy(),
                  atr=np.full(n, atr), rvol=np.zeros(n), gm=np.zeros(n))


def _set(b, i, *, high=None, low=None, open_=None):
    if high is not None:
        b.high[i] = high
    if low is not None:
        b.low[i] = low
    if open_ is not None:
        b.open[i] = open_


E = 10  # entry bar


# (a) target touch
def test_touch_is_first_high_at_or_above_target():
    b = _bars()
    _set(b, E + 5, high=102.0)
    _set(b, E + 9, high=103.0)
    m = P.path_metrics(b, E, 100.0, BASE, gain=0.02, horizon=48, end=len(b.open) - 1)
    assert m["touch"] == E + 5


def test_no_touch_within_horizon_is_none():
    b = _bars()
    _set(b, E + 48, high=105.0)  # 49th bar: outside a 48-bar horizon
    m = P.path_metrics(b, E, 100.0, BASE, gain=0.02, horizon=48, end=len(b.open) - 1)
    assert m["touch"] is None


# (b) MAE before touch
def test_mae_is_lowest_low_strictly_before_touch():
    b = _bars()
    _set(b, E + 2, low=98.5)
    _set(b, E + 6, high=102.5, low=97.0)  # touch bar's own low does not count
    m = P.path_metrics(b, E, 100.0, BASE, gain=0.02, horizon=48, end=len(b.open) - 1)
    assert m["touch"] == E + 6 and m["mae"] == pytest.approx(-0.015)


# (c) ambiguous same-candle touch and stop breach
def test_touch_and_stop_breach_on_one_candle_is_ambiguous():
    b = _bars()
    _set(b, E + GRACE_BARS + 2, high=102.5, low=95.0)  # post-grace stop is -4% (96)
    m = P.path_metrics(b, E, 100.0, BASE, gain=0.02, horizon=48, end=len(b.open) - 1)
    assert m["ambiguous"] is True


# (d) stop in force: grace level, then the tightest entry-anchored ATR stop (ratchet)
def test_stop_path_grace_then_ratchet():
    b = _bars(atr=0.012)  # -4 x 0.012 = -0.048
    b.atr[E + GRACE_BARS + 3:] = 0.0105  # -0.042: tighter, taken
    b.atr[E + GRACE_BARS + 6:] = 0.014  # -0.056 looser: ignored (freqtrade only tightens)
    s = P.stop_path(b, E, 100.0, BASE, P.Rules(), end=E + GRACE_BARS + 10)
    assert s[E] == pytest.approx(73.0)
    assert s[E + GRACE_BARS - 1] == pytest.approx(73.0)
    # first post-grace bar reads the LAST CLOSED candle's ATR (k-1)
    assert s[E + GRACE_BARS] == pytest.approx(100.0 * (1 - 0.048))
    assert s[E + GRACE_BARS + 4] == pytest.approx(100.0 * (1 - 0.042))
    assert s[E + GRACE_BARS + 8] == pytest.approx(100.0 * (1 - 0.042))


# (e) breach bar
def test_breach_is_first_low_at_or_below_stop_in_force():
    b = _bars()
    _set(b, E + 3, low=80.0)  # inside grace: -27% not breached
    _set(b, E + GRACE_BARS + 4, low=95.5)  # post-grace -4% = 96: breached
    m = P.path_metrics(b, E, 100.0, BASE, gain=0.02, horizon=48, end=len(b.open) - 1)
    assert m["breach"] == E + GRACE_BARS + 4 and m["breach_after_grace"] is True


# (f) windows in bars
def test_max_hold_is_960_bars():
    b = _bars(n=2000)
    r = P.replay(b, E, 100.0, BASE, roi=None, roi_suppressed=False, end=1999)
    assert r.reason == "max_hold" and r.idx == E + 960


# (g) replays per coupling
def test_c1_deeper_stop_exits_later_or_not_at_all():
    b = _bars()
    _set(b, E + GRACE_BARS + 4, low=95.5)
    base = P.replay(b, E, 100.0, BASE, roi=None, roi_suppressed=False, end=600)
    c1 = P.replay(b, E, 100.0, BASE, roi=None, roi_suppressed=False, end=600,
                  rules=P.Rules(stop_min_depth=-0.05))
    assert base.reason == "stop" and base.idx == E + GRACE_BARS + 4
    assert c1.idx > base.idx


def test_c3_close_exits_at_bar_49_open():
    b = _bars()
    r = P.replay(b, E, 100.0, BASE, roi=None, roi_suppressed=False, end=600,
                 rules=P.Rules(horizon_action="close"))
    assert r.reason == "horizon_close" and r.idx == E + 48


def test_c3_breakeven_moves_stop_to_entry_from_bar_49():
    b = _bars()
    b.open[E + 40:], b.high[E + 40:], b.low[E + 40:] = 100.5, 100.6, 100.4  # above entry
    _set(b, E + 50, low=99.5)
    r = P.replay(b, E, 100.0, BASE, roi=None, roi_suppressed=False, end=600,
                 rules=P.Rules(horizon_action="breakeven"))
    assert r.reason == "stop" and r.idx == E + 50 and r.price == pytest.approx(100.0)


def test_c5_lock_activates_after_the_touch_candle_not_on_it():
    b = _bars()
    _set(b, E + 30, high=102.5, low=99.0)  # touch, and a low below entry on the SAME candle
    b.open[E + 31:], b.high[E + 31:], b.low[E + 31:] = 101.0, 101.1, 100.9
    _set(b, E + 32, low=99.5)
    r = P.replay(b, E, 100.0, BASE, roi=None, roi_suppressed=False, end=600,
                 rules=P.Rules(lock_after_target=0.0))
    assert r.reason == "stop" and r.idx == E + 32  # not E+30


def test_c4_grace_extends_to_48_bars_at_given_depth():
    b = _bars()
    _set(b, E + GRACE_BARS + 4, low=95.5)  # would breach -4% after 6.5 h
    r = P.replay(b, E, 100.0, BASE, roi=None, roi_suppressed=False, end=600,
                 rules=P.Rules(grace_hours=12.0, grace_level=-0.10))
    assert r.idx > E + GRACE_BARS + 4
    _set(b, E + 20, low=89.0)  # breaches the -10% grace depth
    r = P.replay(b, E, 100.0, BASE, roi=None, roi_suppressed=False, end=600,
                 rules=P.Rules(grace_hours=12.0, grace_level=-0.10))
    assert r.reason == "stop" and r.idx == E + 20


def test_c2_take_profit_from_signal_bar_atr_clamped():
    params = dataclasses.replace(BASE, take_profit=0.022)
    b = _bars(atr=0.002)
    b.atr[E - 1] = 0.004  # signal bar
    # k = 5 -> 0.02 net; open must clear it
    b.open[E + 40:] = 100.0 * 1.0225
    b.high[E + 40:] = b.open[E + 40:] * 1.001
    b.low[E + 40:] = b.open[E + 40:] * 0.999
    r = P.replay(b, E, 100.0, params, roi=None, roi_suppressed=False, end=600,
                 rules=P.Rules(take_profit_atr_mult=5.0))
    assert r.reason == "take_profit" and r.idx == E + 40
    assert P.c2_take_profit(5.0, 0.0001) == 0.01 and P.c2_take_profit(5.0, 0.1) == 0.05


# (h) fees
def test_replay_profit_is_net_of_fees():
    b = _bars()
    _set(b, E + GRACE_BARS + 4, low=95.5)
    r = P.replay(b, E, 100.0, BASE, roi=None, roi_suppressed=False, end=600)
    assert r.profit == pytest.approx(net_from_price_ratio(0.96, BASE.fee_per_side))


# (i) per-bar order
def test_custom_exit_on_open_beats_the_stop_on_the_same_bar():
    params = dataclasses.replace(BASE, take_profit=0.01)
    b = _bars()
    _set(b, E + 30, open_=101.5, high=101.6, low=90.0)
    r = P.replay(b, E, 100.0, params, roi=None, roi_suppressed=False, end=600)
    assert r.reason == "take_profit" and r.idx == E + 30


def test_discretionary_veto_in_first_hour_falls_through_to_stop():
    params = dataclasses.replace(BASE, take_profit=0.001, stoploss_grace_hours=0.0)
    b = _bars()
    _set(b, E + 1, open_=101.0, high=101.0, low=90.0)
    r = P.replay(b, E, 100.0, params, roi=None, roi_suppressed=False, end=600)
    assert r.reason == "stop" and r.idx == E + 1


def test_roi_on_high_fills_at_target_and_is_off_when_suppressed():
    b = _bars()
    _set(b, E + 10, high=106.0)
    r = P.replay(b, E, 100.0, BASE, roi=0.05, roi_suppressed=False, end=600)
    assert r.reason == "roi" and r.profit == pytest.approx(0.05)
    r = P.replay(b, E, 100.0, BASE, roi=0.05, roi_suppressed=True, end=600)
    assert r.reason != "roi"


def test_unclog_uses_the_effective_hours_given():
    params = dataclasses.replace(BASE, unclog_12h_hours=24.0)
    b = _bars()
    b.open[E + 49:] = 101.0
    r = P.replay(b, E, 100.0, params, roi=None, roi_suppressed=False, end=600)
    assert r.reason == "unclog_12h" and r.idx == E + 96


def test_end_of_fold_force_exits_at_close():
    b = _bars()
    r = P.replay(b, E, 100.0, BASE, roi=None, roi_suppressed=False, end=E + 20)
    assert r.reason == "force_exit" and r.idx == E + 21 and r.price == b.close[E + 20]


# (j) no exit threshold literal in paths.py
def test_paths_has_no_exit_threshold_literals():
    src = inspect.getsource(P)
    for lit in ("0.022", "0.27", "0.04", "0.06", "0.005", "960", "6.5"):
        assert lit not in src, lit


# freqtrade rounds a long's stop UP to the price tick (Trade.adjust_stop_loss)
def test_stop_rounds_up_to_tick_and_flat_candle_low_triggers_it():
    assert P.round_up(41.454, 0.1) == pytest.approx(41.5)
    assert P.round_up(41.5, 0.1) == pytest.approx(41.5)
    assert P.tick_size(np.array([41.9, 41.6, 41.5, 41.9])) == pytest.approx(0.1)
    b = _bars(price=42.3)
    b.open[E + 40:], b.high[E + 40:], b.low[E + 40:] = 41.9, 41.9, 41.9
    b.open[E + 41], b.high[E + 41], b.low[E + 41] = 41.6, 41.6, 41.5
    params = dataclasses.replace(BASE, atr_stoploss_cap=-0.02, atr_stoploss_floor=-0.02)
    r = P.replay(b, E, 42.3, params, roi=None, roi_suppressed=False, end=600)
    assert r.idx != E + 41  # 42.3 x 0.98 = 41.454: not touched without rounding
    r = P.replay(b, E, 42.3, params, roi=None, roi_suppressed=False, end=600, tick=0.1)
    assert r.reason == "stop" and r.idx == E + 41 and r.price == pytest.approx(41.5)


def test_roi_suppression_can_be_per_bar():
    b = _bars()
    _set(b, E + 10, high=106.0)
    _set(b, E + 12, high=106.0)
    r = P.replay(b, E, 100.0, BASE, roi=0.05, roi_suppressed={E + 10}, end=600)
    assert r.reason == "roi" and r.idx == E + 12
