"""Per-trade price-path measurements and exit replays (feature 027, research R2/R4, contract C1).

Pure numpy, no freqtrade. Every exit RULE comes from `utils.ExitPolicy` -- this module only orders
the calls the way a freqtrade backtest does on each 15m bar `k` of an open trade:

  1. the stop is updated by `custom_stoploss`, which a backtest calls with the candle's HIGH as
     the current rate (`IStrategy.ft_stoploss_adjust`: `current_rate=(bound or current_rate)`),
     and only while the stop is still below the candle's low (`dir_correct`); the strategy reads
     the LAST CLOSED candle, k-1. Grace level while the trade is younger than the grace window,
     then the entry-anchored `atr_stop_fraction`, kept only when it TIGHTENS (freqtrade never
     loosens); with no grace window the stop is set once, at entry. A wanted stop at or above
     that rate is placed just under it (`STOP_AT_MARKET_RATIO`), as the live code does. Every
     stop price is rounded UP to the pair's price tick (`Trade.adjust_stop_loss`,
     `price_to_precision(..., ROUND_UP)` for a long) when `tick` is given -- on coarse-tick pairs
     that rounding decides whether a flat candle's low triggers the stop;
  2. `custom_exit_decision` on open[k] (guard_metric of k-1), subject to the 1-hour
     discretionary veto -- first exit that fills wins;
  3. the stop, triggered on low[k]: filled at the stop, or at open[k] when the stop is above
     high[k] (`Backtesting._get_close_rate_for_stoploss`);
  4. the native ROI, detected on high[k] and filled at its target (`_get_close_rate_for_roi`),
     also subject to the veto; skipped on bars where `ignore_roi_if_entry_signal` suppressed it
     (`roi_suppressed`: True for every bar, or the set of suppressed bar indices -- the entry
     signal itself is not replayed);
  5. at the fold's last bar `end` an open trade is force-exited at its close; freqtrade stamps
     that exit one bar later (close_date = the next bar's date), so its index is `end + 1`.

`Rules` holds the 027 coupling hooks (C1-C5). All None = today's exit. They are inputs here, NOT
`ExitPolicy` changes: US1 must not edit the framework. The target touch that C3/C5 depend on is
read from CLOSED candles only (k-1 and earlier), never the current candle's high (research R4).

Counterfactual replays ignore re-sequencing (023 Addendum 7): they choose, arms judge.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from utils.ExitPolicy import (
    STOP_AT_MARKET_RATIO,
    ExitParams,
    atr_take_profit,
    effective_stop_fraction,
    custom_exit_decision,
    discretionary_exit_allowed,
    net_from_price_ratio,
    price_ratio_for_net,
)



CLOSE_STOP, CLOSE_ROI, CLOSE_FORCE, CLOSE_HORIZON = "stop", "roi", "force_exit", "horizon_close"

#: Price comparisons at a tick-rounded level: freqtrade compares exact grid prices, a float product
#: of the grid can land 1e-15 off it.
_EPS = 1e-9


def _at_or_below(price: float, level: float) -> bool:
    return price <= level * (1.0 + _EPS)


@dataclass
class Bars:
    """One pair's 15m arrays, aligned by index. `atr` = atr_pct_roll, `gm` = guard_metric."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    rvol: np.ndarray
    gm: np.ndarray


@dataclass(frozen=True)
class Rules:
    """027 coupling hooks; None = off."""

    stop_min_depth: float | None = None  # C1: post-grace stop no tighter than this
    take_profit_atr_mult: float | None = None  # C2: TP = k x signal-bar atr, clamped
    horizon_action: str | None = None  # C3: "close" | "breakeven"
    grace_hours: float | None = None  # C4: grace length
    grace_level: float | None = None  # C4: grace depth
    lock_after_target: float | None = None  # C5: lambda
    target_gain: float = 0.02
    horizon_bars: int = 48


@dataclass(frozen=True)
class Exit:
    idx: int
    price: float
    profit: float  # net of fees
    reason: str


def c2_take_profit(k: float, atr_signal: float) -> float:
    return atr_take_profit(k, atr_signal)


def _grace(params: ExitParams, rules: Rules) -> tuple[float, float]:
    hours = rules.grace_hours if rules.grace_hours is not None else params.stoploss_grace_hours
    level = rules.grace_level if rules.grace_level is not None else params.stoploss_grace_level
    return float(hours or 0.0), float(level if level is not None else 0.0)


def _fraction(b: Bars, i: int, age_h: float, params: ExitParams, rules: Rules) -> float:
    """The stop fraction the strategy wants at `age_h`, from bar i's inputs: the grace level
    inside the window, else the ATR stop -- `ExitPolicy.effective_stop_fraction`, the one copy
    of that choice -- with C4's grace overrides and C1's depth floor applied. C1's `min` cannot
    touch the grace level (always deeper than any C1 depth)."""
    grace_h, grace_lvl = _grace(params, rules)
    if (grace_h, grace_lvl) != (params.stoploss_grace_hours, params.stoploss_grace_level):
        params = dataclasses.replace(params, stoploss_grace_hours=grace_h,
                                     stoploss_grace_level=grace_lvl)
    rv = b.rvol[i] if params.use_volume_confirmation_stoploss else None
    stop = effective_stop_fraction(age_h, float(b.atr[i]), None if rv is None else float(rv),
                                   params)
    if rules.stop_min_depth is not None:
        stop = min(stop, rules.stop_min_depth)
    return stop


def round_up(price: float, tick: float | None) -> float:
    """`price_to_precision(price, ROUND_UP)` on a tick grid (no-op without a tick)."""
    if not tick:
        return price
    return float(np.round(np.ceil(price / tick - 1e-9) * tick, 12))


def tick_size(prices: np.ndarray) -> float | None:
    """The price grid of a window: the smallest gap between distinct prices."""
    u = np.unique(np.round(prices[np.isfinite(prices)], 12))
    gaps = np.diff(u)
    gaps = gaps[gaps > 0]
    return float(gaps.min()) if len(gaps) else None


def _place(wanted: float, rate: float, tick: float | None = None) -> float:
    """The stop price freqtrade ends up with when the strategy wants `wanted`."""
    return round_up(rate * (1.0 + STOP_AT_MARKET_RATIO) if wanted >= rate else wanted, tick)


def first_touch(b: Bars, e: int, end: int, target_price: float) -> int | None:
    """First bar in [e, end] whose high reaches `target_price`."""
    hits = np.nonzero(b.high[e:end + 1] >= target_price)[0]
    return int(e + hits[0]) if len(hits) else None


def _touched(ft: int | None, k: int) -> bool:
    """Target touched on a CLOSED candle before bar k (research R4)."""
    return ft is not None and ft <= k - 1


def _lock(k, e, open_rate, rules: Rules, ft) -> float | None:
    lock = None
    if rules.lock_after_target is not None and _touched(ft, k):
        lock = open_rate * (1.0 + rules.lock_after_target * rules.target_gain)
    if (rules.horizon_action == "breakeven" and k - e >= rules.horizon_bars
            and not _touched(ft, k)):
        lock = open_rate if lock is None else max(lock, open_rate)
    return lock


def _stop_update(b, e, k, open_rate, params, rules, stop, ft=None, tick=None):
    """Stop price in force on bar k (before the bar's low is checked)."""
    age_h = (k - e) * params.timeframe_minutes / 60.0
    if k == e:  # after_fill: the grace level if a window is set, else the ATR stop
        return round_up(open_rate * (1.0 + _fraction(b, k - 1, 0.0, params, rules)), tick)
    if _at_or_below(float(b.low[k]), stop):  # dir_correct: already breached; no adjustment
        return stop
    rate = float(b.high[k])
    # freqtrade CALL SCHEDULE (BaseStrategy.custom_stoploss), not stop arithmetic: without a
    # grace window the stop is fixed at fill; with one, it ratchets to the normal stop after it.
    grace_h, _ = _grace(params, rules)
    if grace_h > 0.0 and age_h >= grace_h:
        wanted = open_rate * (1.0 + _fraction(b, k - 1, age_h, params, rules))
        stop = max(stop, _place(wanted, rate, tick))
    lock = _lock(k, e, open_rate, rules, ft)
    if lock is not None:
        stop = max(stop, _place(lock, rate, tick))
    return stop


def stop_path(b: Bars, e: int, open_rate: float, params: ExitParams, rules: Rules,
              end: int, tick: float | None = None) -> np.ndarray:
    """Stop price in force on every bar e..end, ignoring exits (NaN outside)."""
    out = np.full(len(b.open), np.nan)
    stop = 0.0
    ft = first_touch(b, e, end, open_rate * (1.0 + rules.target_gain))
    for k in range(e, end + 1):
        stop = _stop_update(b, e, k, open_rate, params, rules, stop, ft, tick)
        out[k] = stop
    return out


def path_metrics(b: Bars, e: int, open_rate: float, params: ExitParams, *, gain: float,
                 horizon: int, end: int, tick: float | None = None) -> dict:
    """Target touch within `horizon` bars of entry, MAE before it, ambiguity, first stop breach."""
    last = min(e + horizon - 1, end)
    target = open_rate * (1.0 + gain)
    hits = np.nonzero(b.high[e:last + 1] >= target)[0]
    touch = int(e + hits[0]) if len(hits) else None
    stops = stop_path(b, e, open_rate, params, Rules(), end, tick)
    breached = np.nonzero(b.low[e:end + 1] <= stops[e:end + 1] * (1.0 + _EPS))[0]
    breach = int(e + breached[0]) if len(breached) else None
    grace_h, _ = _grace(params, Rules())
    mae = 0.0
    if touch is not None and touch > e:
        mae = float(b.low[e:touch].min() / open_rate - 1.0)
    return {
        "touch": touch,
        "mae": mae,
        "ambiguous": bool(touch is not None and _at_or_below(float(b.low[touch]), stops[touch])),
        "breach": breach,
        "breach_after_grace": bool(
            breach is not None and (breach - e) * params.timeframe_minutes / 60.0 >= grace_h
        ),
        "stop_at_touch": None if touch is None else float(stops[touch] / open_rate - 1.0),
    }


def replay(b: Bars, e: int, open_rate: float, params: ExitParams, *, roi: float | None,
           roi_suppressed, end: int, rules: Rules | None = None,
           tick: float | None = None) -> Exit:
    """Re-exit one trade entered at open[e] (see module docstring for the per-bar order)."""
    rules = rules if rules is not None else Rules()
    fee = params.fee_per_side
    if rules.take_profit_atr_mult is not None:
        params = dataclasses.replace(
            params, take_profit=c2_take_profit(rules.take_profit_atr_mult, float(b.atr[e - 1])))
    ft = first_touch(b, e, end, open_rate * (1.0 + rules.target_gain))
    stop = 0.0
    for k in range(e, end + 1):
        stop = _stop_update(b, e, k, open_rate, params, rules, stop, ft, tick)
        age_h = (k - e) * params.timeframe_minutes / 60.0
        o = float(b.open[k])
        profit_open = net_from_price_ratio(o / open_rate, fee)
        decision = custom_exit_decision(profit_open, age_h, float(b.gm[k - 1]), params)
        if (rules.horizon_action == "close" and decision in (None, "max_hold")
                and k - e >= rules.horizon_bars and not _touched(ft, k)):
            decision = CLOSE_HORIZON
        if decision is not None and discretionary_exit_allowed(
                decision, age_h, params.min_discretionary_hold_hours):
            return Exit(k, o, profit_open, decision)
        if _at_or_below(float(b.low[k]), stop):
            px = o if stop > b.high[k] * (1.0 + _EPS) else stop
            return Exit(k, px, net_from_price_ratio(px / open_rate, fee), CLOSE_STOP)
        blocked = roi_suppressed is True or (
            not isinstance(roi_suppressed, bool) and k in roi_suppressed)
        if (roi is not None and not blocked
                and net_from_price_ratio(float(b.high[k]) / open_rate, fee) >= roi
                and discretionary_exit_allowed(CLOSE_ROI, age_h,
                                               params.min_discretionary_hold_hours)):
            return Exit(k, open_rate * price_ratio_for_net(roi, fee), roi, CLOSE_ROI)
    c = float(b.close[end])
    return Exit(end + 1, c, net_from_price_ratio(c / open_rate, fee), CLOSE_FORCE)
