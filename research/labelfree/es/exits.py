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

NOTHING IS REPLAYED. Every exit is recomputed:
  unclog_12h / unclog_1d / max_hold / take_profit / the stop stack   purely per-trade
                                                  (BaseStrategy.py:1203-1219), which is what
                                                  makes `hold_window_hours` searchable at all
  model_exit                                      recomputed in panel.py from the side-car's
                                                  three probability columns via the strategy's
                                                  own argmax_with_threshold rule

  The design originally REPLAYED model_exit from the archive and carried an 11.9%
  "invisible to the surrogate" limitation. Regenerating the per-bar probabilities (es/
  signals.py) removed it: the exit signal is now a function of data we hold, so a parameter
  change that shifts when the model exits is visible. REPLAYED_EXIT_SHARE is 0.0 and kept
  only so report.py states the fact rather than silently omitting it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The share of exits this surrogate cannot move. Zero since model_exit became recomputed;
# report.py states the figure rather than omitting it, so a regression here is visible.
REPLAYED_EXIT_SHARE = 0.0
REPLAYED_EXITS: tuple[str, ...] = ()

from utils.ExitPolicy import ExitParams, atr_stop_fraction
VOLUME_TIGHTEN_CLAMP = -0.02          # literal in BaseStrategy.py:1049, not atr_stoploss_cap

# POSITION SIZING. config_wf.json runs a FIXED stake of 900 on a 10,000 wallet, so each trade
# deploys 9% of the INITIAL wallet and returns are ADDITIVE, not compounding. The archive
# confirms it independently: scale_pp_per_abs ~ 0.01 and profit_abs = profit_ratio * 900, so
# one unit of profit_ratio is worth exactly 9 percentage points.
#
# An earlier version compounded prod(1 + pnl) over the pooled trade list, which treats 297
# CONCURRENT trades across 9 pairs as sequential full-wallet bets. On f00 that turned the
# incumbent's real +23.46 pp/yr into -19.06 -- a sign flip, not an offset.
STAKE_PP_PER_RATIO = 900 / 10_000 * 100.0     # = 9.0 pp per unit profit_ratio

# NNNC_MLX.json roi {"0": 0.05}. Not searched, but it FIRES — 2.6% of archived exits — and
# omitting it let take-profit claim trades that ROI would have closed first and higher.
MINIMAL_ROI = 0.05
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
    # Per-bar occupancy for THIS pair. Pooling ORs these across pairs, because the
    # participation floor asks "what fraction of the time is capital deployed" — a
    # PORTFOLIO question. Summing held bars over a summed denominator answers a different
    # one, and answers it ~10x too low on a 10-pair universe: measured on f00, the
    # production incumbent reads 0.011 that way (degenerate, below the 0.05 floor) against
    # 0.0611 as portfolio occupancy (admissible). The shared gate would have rejected
    # PRODUCTION ITSELF.
    occupancy: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.pnl)

    @property
    def mean_exposure(self) -> float:
        """Fraction of bars with a position open. NaN-safe: an empty panel is 0.0, never
        NaN — `nan < floor` is False, so a NaN exposure would pass the participation floor
        silently, which is the exact failure mode the floor exists to catch."""
        if self.occupancy is not None:
            return float(self.occupancy.mean()) if self.occupancy.size else 0.0
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
    def total_pp(self) -> float:
        """Wallet return in percentage points, ADDITIVE — the harness's own convention:
        `wallet_profit_pct = sum(profit_abs) * scale_pp_per_abs` (evaluation/compare_pooled.py
        :29). Fixed stake means a trade's contribution does not depend on wallet history."""
        return STAKE_PP_PER_RATIO * float(np.sum(self.pnl)) if self.pnl else 0.0

    @property
    def annualised_pp(self) -> float:
        """pp per year by SIMPLE division, matching compare_pooled.py:46 (`v / years`)."""
        if not self.pnl or self.span_years <= 0:
            return 0.0
        return self.total_pp / self.span_years

    def block_returns(self, n_blocks: int = 16) -> list[float]:
        """Trade P&L bucketed onto the TIME axis, in pp, for DSR and CSCV.

        DSR deflates a Sharpe computed over time and CSCV partitions that axis into blocks;
        both need a SERIES. Without one, every configuration's column is constant, CSCV finds
        the in-sample and out-of-sample rankings identical in every split, and PBO comes back
        as exactly 0.0 — which is what happened on the first full run here, and is the same
        schema gap that made SC-004 unevaluable in four consecutive features.

        Bucketed by EXIT bar against the pooled bar axis, so a trade contributes to the block
        in which it was realised.
        """
        import numpy as np
        axis = self.occupancy.size if self.occupancy is not None else self.n_bars_total
        if not self.pnl or axis <= 0:
            return []
        edges = np.linspace(0, axis, n_blocks + 1)
        out = np.zeros(n_blocks)
        for x, q in zip(self.exit_bar, self.pnl):
            b = min(int(np.searchsorted(edges, x, side="right") - 1), n_blocks - 1)
            out[max(0, b)] += q
        return [float(v * STAKE_PP_PER_RATIO) for v in out]

    def exit_mix(self) -> dict[str, float]:
        if not self.reason:
            return {}
        n = len(self.reason)
        return {r: self.reason.count(r) / n for r in sorted(set(self.reason))}


def atr_stop_level(atr_pct_roll: float, rvol_stop: float, p: dict) -> float:
    """The entry-relative ATR stop. Delegates to `utils.ExitPolicy` (feature 021 FR-002).

    This function used to re-implement the clamp -- a FOURTH copy of the arithmetic,
    beyond the three feature 021's research R1 catalogued. Its own fidelity notes had
    independently rediscovered that the volume-tighten clamp is a literal -0.02 rather
    than `atr_stoploss_cap`, which is a good illustration of why one copy is better than
    four: two authors each had to find that trap separately.

    Behaviour is unchanged -- `utils.ExitPolicy.atr_stop_fraction` is the same expression
    with the same literal, and the fidelity notes above this function still describe what
    is mirrored. `VOLUME_TIGHTEN_CLAMP` is retained as a module constant because the
    fidelity notes reference it by name.
    """
    return atr_stop_fraction(
        atr_pct_roll,
        rvol_stop,
        ExitParams(
            atr_stoploss_multiplier=p["atr_stoploss_multiplier"],
            atr_stoploss_cap=p["atr_stoploss_cap"],
            atr_stoploss_floor=p["atr_stoploss_floor"],
            static_stoploss=float(p.get("static_stoploss", -0.05)),
            use_atr_adaptive_stoploss=True,
            use_volume_confirmation_stoploss=True,
        ),
    )


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
    roi_level = float(p.get("minimal_roi", MINIMAL_ROI))

    occ = np.zeros(n, dtype=bool)
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
            # FIDELITY (backtesting.py:992-1000, interface.py:1440-1463). freqtrade calls
            # should_exit with rate = row[OPEN], so `current_profit` — what every custom_exit
            # reason tests — is computed from the OPEN. `current_profit_best` comes from the
            # HIGH and is used by ROI ONLY. The stop triggers on the LOW. Three different
            # prices, and using the wrong one biases the two dominant mechanisms.
            profit_open = op / entry_px - 1.0
            profit_high = hi / entry_px - 1.0

            # FIDELITY — THE SEQUENCE (interface.py:1503-1507, verbatim):
            #     # Sequence:
            #     # Exit-signal
            #     # Stoploss
            #     # ROI
            #     # Trailing stoploss
            # The CUSTOM EXIT is appended to `exits` BEFORE the stoploss flag, so a bar on
            # which both fire resolves as the custom exit. Checking the stop first (as an
            # earlier version did) systematically converts winners into stop-outs on exactly
            # the two mechanisms that decide this strategy's P&L.

            # ---- 1. custom_exit — tested on the OPEN, filled at the OPEN ------------------
            #         (interface.py:1472-1478 passes current_rate/current_profit from `rate`,
            #          which should_exit was called with as row[OPEN]; backtesting.py:594-595
            #          fills every non-stoploss/non-ROI exit at row[OPEN].)
            if profit_open > take_profit:
                exit_i, reason, pnl = j, "take_profit", profit_open
                break
            if model_exit is not None and model_exit[j]:
                exit_i, reason, pnl = j, "model_exit", profit_open
                break
            hours = age / BARS_PER_HOUR
            if age >= hold_window_bars and profit_open > 0.005:
                exit_i, reason, pnl = j, "unclog_12h", profit_open
                break
            if hours >= 24.0 and profit_open >= 0.0:
                exit_i, reason, pnl = j, "unclog_1d", profit_open
                break
            if age >= max_bars:
                exit_i, reason, pnl = j, "max_hold", profit_open
                break

            # ---- 2. the stoploss — triggered on the LOW ----------------------------------
            if grace_bars > 0 and age < grace_bars:
                level = grace_lvl                       # wide, anchored to entry, no trail
                in_grace = True
            else:
                cand = atr_stop_level(atr_pct_roll[j], rvol_stop[j], p)
                eff_stop = cand if eff_stop is None else max(eff_stop, cand)  # tighten only
                level = eff_stop
                in_grace = False

            stop_px = entry_px * (1.0 + level)
            if lo <= stop_px:
                exit_i = j
                reason = "stop_loss" if in_grace else "trailing_stop_loss"
                # backtesting.py:614-616 — the OPEN is used ONLY when the stop sits above the
                # candle's HIGH, i.e. the whole candle gapped past it. Otherwise the fill is
                # AT THE STOP.
                px = op if stop_px > hi else stop_px
                pnl = px / entry_px - 1.0
                break

            # ---- 3. ROI — the ONE check tested on the HIGH (interface.py:1443-1446) -------
            # NOT on the fill bar. The trade enters at that bar's OPEN, so triggering on the
            # SAME bar's high assumes the most favourable intrabar path on the one bar we
            # have no path information for. freqtrade special-cases trade_dur == 0 for this
            # reason (backtesting.py:618-621, "most pessimistic price movement"), and taking
            # the conservative branch is the honest approximation. Measured on f07: 6 of 28
            # ROI exits came from the fill bar against a real backtest's 13 ROI exits total.
            if age > 0 and profit_high >= roi_level:
                exit_i, reason, pnl = j, "roi", roi_level
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
        occ[entry_i:exit_i] = True
        i = exit_i + 1                                  # one position at a time, per pair

    ts.occupancy = occ
    return ts
