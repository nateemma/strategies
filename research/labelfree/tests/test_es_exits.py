"""Surrogate fidelity tests (T013-T016).

These are the feature's decisive tests: everything downstream calls `simulate` thousands of
times, so a surrogate that does not reproduce the real exit stack means the search optimises
a fiction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.es import exits as EX          # noqa: E402
from research.labelfree.es import space as SP          # noqa: E402

BPH = EX.BARS_PER_HOUR


def _bars(n, start=100.0, drift=0.0, vol=0.0, seed=7):
    """A deterministic price path with explicit high/low."""
    rng = np.random.default_rng(seed)
    steps = drift + vol * rng.standard_normal(n)
    px = start * np.cumprod(1.0 + steps)
    return px


def _panel(px, atr=0.010, rvol=1.0, spread=0.0):
    n = len(px)
    return dict(open_px=px,
                high=px * (1.0 + spread),
                low=px * (1.0 - spread),
                atr_pct_roll=np.full(n, atr),
                rvol_stop=np.full(n, rvol),
                cost=np.zeros(n))


def _params(**over):
    p = dict(SP.INCUMBENT)
    p.update(over)
    return p


# --------------------------------------------------------------- T014: parameters must bite

def test_parameters_change_the_trade_series():
    """FR-001. The test brief 1's simulator could never have passed.

    Brief 1's `sim.replay()` takes the trade series as an INPUT, so its trades are identical
    under every parameter set by construction. If this test ever passes trivially, the
    surrogate has regressed to that.
    """
    n = 4000
    px = _bars(n, drift=0.0, vol=0.004)
    pan = _panel(px, spread=0.002)
    entries = np.zeros(n, bool)
    entries[::200] = True

    tight = EX.simulate(entries, params=_params(cexit_take_profit=0.010), **pan)
    loose = EX.simulate(entries, params=_params(cexit_take_profit=0.060), **pan)

    assert len(tight) > 0 and len(loose) > 0, "no trades — the test cannot discriminate"
    assert (tight.exit_bar != loose.exit_bar) or (tight.pnl != loose.pnl), \
        "take-profit did not change the trade series: the surrogate ignores its parameters"
    assert tight.mean_hold_hours < loose.mean_hold_hours, \
        "a tighter take-profit must exit sooner on average"


def test_stop_multiplier_changes_the_trade_series():
    """The brief's headline lever must reach the trades (Principle IV, applied here)."""
    n = 4000
    px = _bars(n, drift=-0.0004, vol=0.003)
    pan = _panel(px, atr=0.012, spread=0.002)
    entries = np.zeros(n, bool)
    entries[::200] = True

    # grace 0 so the ATR stop is reachable from the first bar
    near = EX.simulate(entries, params=_params(stoploss_grace_hours=0.0,
                                               atr_stoploss_multiplier=1.0), **pan)
    far = EX.simulate(entries, params=_params(stoploss_grace_hours=0.0,
                                              atr_stoploss_multiplier=5.0,
                                              atr_stoploss_floor=-0.10), **pan)
    assert near.exit_bar != far.exit_bar, "atr_stoploss_multiplier is inert in the surrogate"


# ------------------------------------------------------------------ T015: grace gates the ATR

def test_grace_gates_the_stop():
    """Fidelity note 1: the grace branch RETURNS EARLY, so it gates the ATR stop.

    Inside the window the stop sits at `stoploss_grace_level`, anchored to entry, and does
    not trail. A 6% drawdown must therefore NOT stop out under a -0.27 grace level, even
    though the ATR stop (-0.025 here) would have fired immediately.
    """
    n = 200
    px = np.full(n, 100.0)
    px[10:] = 94.0                                  # -6% immediately after entry, then flat
    pan = _panel(px, atr=0.010)                     # ATR stop = -2.5 * 0.010 = -0.025
    entries = np.zeros(n, bool)
    entries[0] = True

    graced = EX.simulate(entries, params=_params(stoploss_grace_hours=6.5,
                                                 stoploss_grace_level=-0.27,
                                                 cexit_take_profit=0.99), **pan)
    grace_bars = int(round(6.5 * BPH))
    age_at_exit = graced.exit_bar[0] - graced.entry_bar[0]
    assert age_at_exit >= grace_bars, (
        f"the ATR stop fired {age_at_exit} bars in, inside the {grace_bars}-bar grace "
        f"window — the grace branch is not gating it")
    assert graced.reason[0] == "trailing_stop_loss", (
        "once the window expires the ATR stop must fire; grace DELAYS it, it does not "
        "remove it (BaseStrategy.py:1081-1086 tightens to _resolve_normal_stop)")

    ungraced = EX.simulate(entries, params=_params(stoploss_grace_hours=0.0,
                                                   cexit_take_profit=0.99), **pan)
    assert ungraced.reason[0] == "trailing_stop_loss", \
        "with grace off the ATR stop must fire on a -6% move against a -0.025 stop"
    assert ungraced.exit_bar[0] < graced.exit_bar[0], \
        "removing the grace window must exit strictly sooner"


def test_grace_stop_still_fires_when_hit():
    """The grace stop is WIDE, not absent — BaseStrategy's own comment says so."""
    n = 200
    px = np.full(n, 100.0)
    px[10:] = 60.0                                  # -40%, through a -0.27 grace level
    pan = _panel(px, atr=0.010)
    entries = np.zeros(n, bool)
    entries[0] = True
    ts = EX.simulate(entries, params=_params(stoploss_grace_hours=6.5,
                                             stoploss_grace_level=-0.27,
                                             cexit_take_profit=0.99), **pan)
    assert ts.reason[0] == "stop_loss", "the wide grace stop must still fire when breached"


# ------------------------------------------------- T016: volume-confirmation tightening

@pytest.mark.parametrize("rvol,expect_tighter", [(1.0, False), (4.0, True)])
def test_volume_confirmation_tightening(rvol, expect_tighter):
    """rvol > 1 tightens by 1/sqrt(rvol), clamped at a LITERAL -0.02 (fidelity note 4)."""
    p = _params(atr_stoploss_multiplier=2.5, atr_stoploss_floor=-0.10,
                atr_stoploss_cap=-0.005)
    base = EX.atr_stop_level(0.012, 1.0, p)          # -0.030
    got = EX.atr_stop_level(0.012, rvol, p)
    if expect_tighter:
        assert got > base, "rvol > 1 must tighten the stop (move it closer to zero)"
        # BaseStrategy.py:1049 is `min(stop / rvol**0.5, -0.02)`: the tightening is capped,
        # so -0.02 is as tight as volume confirmation is ever allowed to make it.
        assert got == pytest.approx(min(base / rvol ** 0.5, EX.VOLUME_TIGHTEN_CLAMP))
    else:
        assert got == pytest.approx(base), "rvol <= 1 must leave the stop untouched"


def test_tightening_clamp_is_literal_not_the_cap():
    """The clamp is -0.02 even when atr_stoploss_cap is -0.005. Mirrors the real asymmetry."""
    p = _params(atr_stoploss_multiplier=2.5, atr_stoploss_floor=-0.10,
                atr_stoploss_cap=-0.005)
    assert EX.atr_stop_level(0.012, 100.0, p) == pytest.approx(EX.VOLUME_TIGHTEN_CLAMP)


# ------------------------------------------------------- T019a: time exits are RECOMPUTED

def test_hold_window_defers_unclog():
    """`hold_window_hours` is dimension 10 — it must actually move the unclog exit."""
    n = 600
    px = np.full(n, 100.0)
    px[2:] = 100.8          # step AFTER the fill bar — entry is at bar 1, so a step at
                            # bar 1 would make the entry price 100.8 and the profit 0.0
    pan = _panel(px, atr=0.010)
    entries = np.zeros(n, bool)
    entries[0] = True
    base = _params(stoploss_grace_hours=0.0, cexit_take_profit=0.99,
                   atr_stoploss_floor=-0.10)

    early = EX.simulate(entries, params={**base, "hold_window_hours": 12.0}, **pan)
    late = EX.simulate(entries, params={**base, "hold_window_hours": 48.0}, **pan)
    assert early.reason[0] == "unclog_12h"
    assert late.exit_bar[0] > early.exit_bar[0], \
        "widening hold_window_hours must defer the unclog exit — otherwise dimension 10 is inert"


def test_max_hold_bounds_the_trade():
    n = 4000
    px = np.full(n, 100.0)                          # flat: nothing else can fire
    pan = _panel(px, atr=0.010)
    entries = np.zeros(n, bool)
    entries[0] = True
    ts = EX.simulate(entries, params=_params(stoploss_grace_hours=0.0,
                                             cexit_take_profit=0.99,
                                             cexit_max_days=5), **pan)
    assert ts.reason[0] in ("max_hold", "unclog_1d")
    assert ts.exit_bar[0] - ts.entry_bar[0] <= 5 * EX.BARS_PER_DAY


# --------------------------------------------------------------------- structural invariants

def test_entry_fills_on_the_next_bar():
    """No same-bar fill — the signal bar's own open is not tradeable."""
    n = 100
    px = np.full(n, 100.0)
    pan = _panel(px)
    entries = np.zeros(n, bool)
    entries[10] = True
    ts = EX.simulate(entries, params=_params(stoploss_grace_hours=0.0), **pan)
    assert ts.entry_bar[0] == 11


def test_one_position_at_a_time_per_pair():
    n = 2000
    px = _bars(n, vol=0.003)
    pan = _panel(px, spread=0.002)
    entries = np.ones(n, bool)                       # a signal on every single bar
    ts = EX.simulate(entries, params=_params(), **pan)
    for a, b in zip(ts.exit_bar, ts.entry_bar[1:]):
        assert b > a, "overlapping positions on one pair"
