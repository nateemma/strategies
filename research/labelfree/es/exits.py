"""The exit surrogate (T017, T018, T019, T019a) — the feature's core risk.

WHY THIS EXISTS AND BRIEF 1'S SIMULATOR DOES NOT SUFFICE. `sim.replay()` is documented as
"Replay a FIXED entry/exit series": the trade series is an INPUT. `sim.net_returns` is
exposure-path economics (w·r - c·|Δw|) with no stop, no grace, no holding limit. Every
parameter this feature searches DETERMINES the trade series, so the simulator consumes
exactly what the search is trying to change. And a freqtrade backtest per evaluation, at the
measured ~175 s, would cost ~1,167 hours for the planned budget.

FIDELITY NOTES — each one read out of Framework/BaseStrategy.py, not assumed:

  1. The grace branch RETURNS EARLY (BaseStrategy.py:1069-1086). With grace_hours > 0 -- the
     incumbent is 6.5 -- custom_stoploss NEVER reaches the ATR branch below it. The ATR stop
     is reached only through _resolve_normal_stop after the window expires. So grace GATES
     the ATR stop; it does not merely precede it.

  2. freqtrade ONLY TIGHTENS a stop. After grace expires the strategy returns an
     entry-anchored ATR level on EVERY bar, recomputed from that bar's atr_pct_roll, and
     freqtrade keeps the tightest. The effective stop is therefore the RUNNING MINIMUM of the
     entry-anchored ATR level since grace expiry — not the value at any single bar.

  3. This is also why stop exits are labelled `trailing_stop_loss` rather than `stop_loss`:
     interface.py:1654-1658 upgrades the label whenever the stop that fired differs from the
     initial one. The native trailing stop is OFF (BaseStrategy.trailing_stop = False). Those
     98 of 378 archived exits are this stack, not freqtrade's trailing stop.

  4. The volume-confirmation tightening clamps at a LITERAL -0.02, independent of
     atr_stoploss_cap (BaseStrategy.py:1049). Mirrored exactly, including that asymmetry.

  5. rvol for the stop is volume / volume.rolling(20, min_periods=5).mean() — NOT the `rvol`
     column, which is an ewm(span=10). Different quantity, same name.

WHAT IS REPLAYED AND WHAT IS RECOMPUTED:
  recomputed  unclog_12h, unclog_1d, max_hold, take_profit, the stop stack, the entry guards
              — all purely per-trade (BaseStrategy.py:1203-1219), which is what makes
              `hold_window_hours` searchable at all
  replayed    model_exit only (11.9% of trades). It depends on the model's per-bar
              prediction, which is not a searched parameter. A parameter change that would
              have altered when the model exits is invisible here, and report.py says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The share of archived trades whose exit this surrogate cannot move. report.py states it.
REPLAYED_EXIT_SHARE = 0.119
REPLAYED_EXITS = ("model_exit",)
VOLUME_TIGHTEN_CLAMP = -0.02          # literal in BaseStrategy.py:1049, not atr_stoploss_cap
BARS_PER_HOUR = 4                     # 15-minute bars
BARS_PER_DAY = 96


@dataclass
class TradeSeries:
    entry_bar: list[int] = field(default_factory=list)
    exit_bar: list[int] = field(default_factory=list)
    pnl: list[float] = field(default_factory=list)          # net of costs, fractional
    reason: list[str] = field(default_factory=list)
    n_bars_total: int = 0
    bars_per_year: float = 365.25 * BARS_PER_DAY

    def __len__(self) -> int:
        return len(self.pnl)

    @property
    def mean_exposure(self) -> float:
        if not self.n_bars_total:
            return 0.0
        held = sum(x - e for e, x in zip(self.entry_bar, self.exit_bar))
        return held / self.n_bars_total

    @property
    def mean_hold_hours(self) -> float:
        if not self.pnl:
            return 0.0
        return float(np.mean([(x - e) / BARS_PER_HOUR
                              for e, x in zip(self.entry_bar, self.exit_bar)]))

    @property
    def span_years(self) -> float:
        return self.n_bars_total / self.bars_per_year

    @property
    def annualised_pp(self) -> float:
        """Compounded annualised return in percentage points."""
        if not self.pnl or self.span_years <= 0:
            return 0.0
        equity = float(np.prod([1.0 + p for p in self.pnl]))
        if equity <= 0:
            return -100.0
        return (equity ** (1.0 / self.span_years) - 1.0) * 100.0

    def exit_mix(self) -> dict[str, float]:
        if not self.reason:
            return {}
        n = len(self.reason)
        return {r: self.reason.count(r) / n for r in sorted(set(self.reason))}


def atr_stop_level(atr_pct_roll: float, rvol_stop: float, p: dict) -> float:
    """The entry-relative ATR stop, mirroring BaseStrategy._resolve_normal_stop exactly."""
    if not (atr_pct_roll > 0):
        return float(p.get("static_stoploss", -0.05))
    stop = -p["atr_stoploss_multiplier"] * atr_pct_roll
    stop = max(min(stop, p["atr_stoploss_cap"]), p["atr_stoploss_floor"])
    if rvol_stop > 1.0:
        # Fidelity note 4: the clamp is a literal -0.02, NOT atr_stoploss_cap.
        stop = min(stop / (rvol_stop ** 0.5), VOLUME_TIGHTEN_CLAMP)
    return stop


def simulate(entries, open_px, high, low, atr_pct_roll, rvol_stop, cost, params,
             model_exit=None, static_stoploss: float = -0.05) -> TradeSeries:
    """Replay one pair's bars under a candidate's exit stack.

    `entries` is a boolean per-bar array of admitted entry signals (the guards have already
    been applied by panel.py, since they decide which trades exist at all). `model_exit` is
    the archived per-bar model exit flag, replayed rather than recomputed.
    """
    n = len(open_px)
    ts = TradeSeries(n_bars_total=n)
    p = dict(params, static_stoploss=static_stoploss)
    grace_h = float(p["stoploss_grace_hours"])
    grace_lvl = float(p["stoploss_grace_level"])
    take_profit = float(p["cexit_take_profit"])
    max_bars = int(p["cexit_max_days"]) * BARS_PER_DAY
    hold_window_bars = int(round(float(p["hold_window_hours"]) * BARS_PER_HOUR))
    grace_bars = int(round(grace_h * BARS_PER_HOUR))

    i = 0
    while i < n - 1:
        if not entries[i]:
            i += 1
            continue
        entry_i = i + 1                       # fill on the next bar's open — never this bar
        if entry_i >= n:
            break
        entry_px = open_px[entry_i]
        if not (entry_px > 0):
            i += 1
            continue

        # Fidelity note 2: the effective stop is the running MINIMUM (tightest) of the
        # entry-anchored level, because freqtrade only ever tightens.
        eff_stop = None
        exit_i, reason = None, None

        for j in range(entry_i, min(n, entry_i + max_bars + 1)):
            age = j - entry_i
            hi, lo, op = high[j], low[j], open_px[j]

            # ---- the stop level in force on this bar -------------------------------------
            if grace_bars > 0 and age < grace_bars:
                level = grace_lvl                       # wide, anchored to entry, no trail
            else:
                cand = atr_stop_level(atr_pct_roll[j], rvol_stop[j], p)
                eff_stop = cand if eff_stop is None else max(eff_stop, cand)  # tighten only
                level = eff_stop

            stop_px = entry_px * (1.0 + level)
            if lo <= stop_px:
                exit_i, reason = j, ("stop_loss" if (grace_bars > 0 and age < grace_bars)
                                     else "trailing_stop_loss")
                px = min(op, stop_px)                   # gap-through fills at the open
                pnl = px / entry_px - 1.0
                break

            # ---- take-profit (BaseStrategy.py:1203) --------------------------------------
            tp_px = entry_px * (1.0 + take_profit)
            if hi >= tp_px:
                exit_i, reason = j, "take_profit"
                px = max(op, tp_px)
                pnl = px / entry_px - 1.0
                break

            # ---- replayed model exit -----------------------------------------------------
            if model_exit is not None and model_exit[j]:
                exit_i, reason = j, "model_exit"
                pnl = op / entry_px - 1.0
                break

            # ---- recomputed time exits (BaseStrategy.py:1210-1219) -----------------------
            profit = op / entry_px - 1.0
            hours = age / BARS_PER_HOUR
            if age >= hold_window_bars and profit > 0.005:
                exit_i, reason = j, "unclog_12h"
                pnl = profit
                break
            if hours >= 24.0 and profit >= 0.0:
                exit_i, reason = j, "unclog_1d"
                pnl = profit
                break
            if age >= max_bars:
                exit_i, reason = j, "max_hold"
                pnl = profit
                break

        if exit_i is None:                              # ran off the end of the range
            exit_i = min(n - 1, entry_i + max_bars)
            reason = "force_exit"
            pnl = open_px[exit_i] / entry_px - 1.0

        # round-trip cost, per side, from the brief-1 cost model
        pnl -= float(cost[entry_i]) + float(cost[exit_i])
        ts.entry_bar.append(entry_i)
        ts.exit_bar.append(exit_i)
        ts.pnl.append(float(pnl))
        ts.reason.append(reason)
        i = exit_i + 1                                  # one position at a time, per pair

    return ts
