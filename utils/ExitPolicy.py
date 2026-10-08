# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""The SINGLE source of truth for the exit stop arithmetic (feature 021, FR-001/FR-002).

Before this module the same computation lived in THREE places:

  1. `Framework.BaseStrategy.custom_stoploss`      (the `after_fill` ATR branch)
  2. `Framework.BaseStrategy._resolve_normal_stop` (whose docstring admitted it
                                                    "Mirrors the after_fill ATR logic")
  3. `NNPredict.NNPredictStrategy._stop_aware_labels` (whose docstring admitted it used
                                                    "the SAME stop arithmetic")

Three copies of one rule is how a label ends up reasoning about an exit the strategy
does not run. All three are now callers.

AMENDED 2026-10-08 (T021 follow-up, FR-001's amendment). T021's reconciliation found the
first outcome table optimistic by 2.70pp win-rate / 10.7% mean-profit-per-trade because
`simulate_forward` modelled only stop / take-profit / hold-cap, while the LIVE
`BaseStrategy.custom_exit` also fires on `metric_overbought`, `unclog_12h`, `unclog_1d` and
`max_hold`, and freqtrade itself ALSO applies a native ROI exit (`minimal_roi`) that
`custom_exit` never sees. `custom_exit_decision` below is the single source of truth for the
first group (`BaseStrategy.custom_exit` is now a caller of it); `simulate_forward` calls it
too, so the OFFLINE and LIVE decisions for these exits cannot drift apart. The native ROI
exit has no live-side equivalent to extract (freqtrade's own ROI engine evaluates it, not
strategy code), so it is modelled only in `simulate_forward`, gated by `ExitParams.roi`.

REAL PRECEDENCE, verified against freqtrade source rather than assumed (see
`research/labelfree/es/exits.py`'s independent fidelity notes for a second, prior-art
confirmation of the same facts, concerning a different feature):

  * `freqtrade/strategy/interface.py::should_exit` appends the custom-exit outcome to its
    `exits` list BEFORE the native stoploss, and the native ROI after that
    ("# Sequence: Exit-signal / Stoploss / ROI / Trailing stoploss", interface.py:1504-1507).
  * `freqtrade/optimize/backtesting.py::_check_trade_exit` iterates that list and returns on
    the FIRST exit that fills, so custom_exit's decision wins any same-bar tie against the
    native stop, which in turn wins against ROI.
  * `should_exit` calls `custom_exit` with `rate = row[OPEN_IDX]` (backtesting.py:992-1000),
    so EVERY decision inside `custom_exit` -- `metric_overbought`, `take_profit`,
    `unclog_12h`, `unclog_1d`, `max_hold` -- is a function of the bar's OPEN price, not its
    high or low. The native stop triggers on the LOW; the native ROI is the one check
    evaluated against the HIGH (`current_profit_best`, interface.py:1443-1446).

WHAT THIS MODULE DOES AND DOES NOT CHANGE ABOUT THE EXISTING stop/take-profit PAIR.
`simulate_forward`'s pre-existing `stop` (low-based) vs `take_profit` (high-based) same-bar
tie-break ("stop wins", contracts/exit-policy.md) is UNCHANGED and still the only mechanism
used when a caller does not supply `open_`. Once `open_` IS supplied (the real per-bar outcome
table always will), `custom_exit_decision` resolves `take_profit` correctly from the OPEN
price, BEFORE the stop check, matching the bullet above -- and the legacy high-based
`take_profit` check is skipped entirely in that case, because firing it would re-introduce
exactly the kind of phantom, OPEN-blind profit T021 found. **This is the one place in this
amendment that changes decision MECHANICS rather than purely adding new exit types**, and it
is deliberate: once the correct, source-grounded mechanism is available, leaving the old
high-based approximation active alongside it would silently resurrect optimism on the bars
where they disagree. See the handback report for the residual uncertainty this still carries.

DELIBERATELY PURE. No freqtrade import, no `Trade`, no dataprovider. That is what lets
offline label construction ask "what would the exit have done here?" without
instantiating a strategy -- and it is why this lives in `utils/` rather than
`Framework/` (strategy-agnostic *runtime* code) or `evaluation/` (nothing there runs
inside a backtest, so the live path must not depend on it).

What does NOT live here: freqtrade's return convention. `custom_stoploss` returning
`1.0` to mean "no change", and its rescaling of the post-grace stop relative to
`current_rate`, are glue whose meaning depends on how freqtrade interprets the value.
They stay in `BaseStrategy`.

Guarded by two tests: `test_exit_policy_characterisation.py` proves the live decision is
unchanged bar-by-bar, and `test_exit_policy_single_source.py` fails if the arithmetic
reappears anywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Hard floor applied AFTER volume confirmation tightens the stop.
#:
#: This is a LITERAL in the original code, and it is only coincidentally equal to the
#: default `atr_stoploss_cap` (-0.02). Substituting the cap here would be a silent
#: behaviour change for anyone who retunes the cap, so it stays a separate constant.
#: Verified against the characterisation fixture.
VOLUME_CONFIRMED_STOP_FLOOR = -0.02


@dataclass(frozen=True)
class ExitParams:
    """The parameters the stop arithmetic reads. Values come from the strategy.

    Frozen because an exit configuration identifies a set of outcomes: feature 021
    keys its outcome table on a hash of these, so a mutable config would let a cached
    table silently describe a different exit.
    """

    atr_stoploss_multiplier: float = 2.5
    atr_stoploss_cap: float = -0.02      # tightest allowed (closest to zero)
    atr_stoploss_floor: float = -0.04    # loosest allowed (most negative)
    static_stoploss: float = -0.05
    use_atr_adaptive_stoploss: bool = False
    use_volume_confirmation_stoploss: bool = True
    #: NO DEFAULT, deliberately. Three different grace configurations exist in this
    #: repository -- BaseStrategy.sell_params says 2.6h/-0.17, NNNC_MLX_Open.json (the
    #: production operating point) says 6.5h/-0.27, and a dataclass default of "no grace"
    #: would be a third. An offline caller that forgot to supply these would silently
    #: simulate a stop firing inside a window where production has effectively none
    #: (-27% for 6.5 hours), and every outcome would describe an exit the strategy does
    #: not run. `effective_stop_fraction` raises rather than guessing.
    #:
    #: The LIVE path never reads these: `custom_stoploss` handles grace itself and returns
    #: before consulting ExitParams, and `_resolve_normal_stop` is only reached post-grace.
    stoploss_grace_hours: float | None = None
    stoploss_grace_level: float | None = None
    #: Forward-simulation fields. Unused by the live stop path (freqtrade owns the
    #: take-profit and the hold cap); required by `simulate_forward`, which has to close
    #: a position the way the strategy actually would.
    take_profit: float = 0.022
    max_hold_days: int = 10
    fee_per_side: float = 0.001
    timeframe_minutes: int = 15

    #: FR-001's amendment -- the rest of `BaseStrategy.custom_exit`'s decision, plus the
    #: native ROI exit it never sees. All read the SAME `current_profit`/`age_hours` inputs
    #: `custom_exit_decision` is called with; see this module's docstring for why that is the
    #: OPEN price live, not the high or low.
    #:
    #: Fixed literals in `BaseStrategy.custom_exit` -- not freqtrade `Parameter`s, verified by
    #: a repo-wide grep finding exactly one occurrence of each -- so, unlike the grace fields,
    #: a default matching the live literal is safe rather than a silent guess.
    metric_overbought_threshold: float = 0.98
    unclog_12h_hours: float = 12.0
    unclog_12h_min_profit: float = 0.005
    unclog_1d_hours: float = 24.0
    #: Whether the metric_overbought/take_profit profit-check block runs at all
    #: (`cexit_enable_profit_checks`, a freqtrade CategoricalParameter, default True, True at
    #: the Open operating point). `unclog_12h`/`unclog_1d`/`max_hold` are NOT gated by this in
    #: the live code -- they are evaluated unconditionally, after the gated block.
    cexit_enable_profit_checks: bool = True
    #: The native `minimal_roi` level (`BaseStrategy` 0.03, `NNNCStrategy` 0.05 -- see this
    #: module's docstring). `None` means "do not model the ROI exit" (the live default
    #: for `BaseStrategy.custom_exit`'s own callers, which never need it: freqtrade's ROI
    #: engine, not strategy code, owns this check, so there is nothing for `custom_exit` to
    #: delegate to and this field exists only for `simulate_forward`).
    roi: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.atr_stoploss_floor <= self.atr_stoploss_cap < 0:
            raise ValueError(
                "require atr_stoploss_floor <= atr_stoploss_cap < 0, got "
                f"floor={self.atr_stoploss_floor} cap={self.atr_stoploss_cap}"
            )
        if not self.take_profit > 0:
            raise ValueError(f"require take_profit > 0, got {self.take_profit}")
        if not self.max_hold_days >= 1:
            raise ValueError(f"require max_hold_days >= 1, got {self.max_hold_days}")
        if self.fee_per_side < 0:
            raise ValueError(f"require fee_per_side >= 0, got {self.fee_per_side}")
        if self.roi is not None and not self.roi > 0:
            raise ValueError(f"require roi > 0 when set, got {self.roi}")


def atr_stop_fraction_array(atr_pct, rvol, params: ExitParams):
    """Array form of the stop level -- THE one clamp expression in the codebase.

    `atr_stop_fraction` is a thin scalar wrapper over this, so the arithmetic exists
    once even though it is consumed both per-trade (live, scalar) and per-bar
    (offline label construction, vectorised over ~200k bars per pair).

    `rvol=None` means "do not apply volume confirmation". That is not a convenience:
    `NNPredict._stop_aware_labels` deliberately omits the volume-confirmation step, and
    passing None makes that approximation an explicit argument rather than a silently
    missing code path.
    """
    import numpy as np

    atr = np.asarray(atr_pct, dtype=float)

    if not params.use_atr_adaptive_stoploss:
        return np.full(atr.shape, params.static_stoploss, dtype=float)

    stop = -params.atr_stoploss_multiplier * atr
    stop = np.maximum(
        np.minimum(stop, params.atr_stoploss_cap), params.atr_stoploss_floor
    )

    if params.use_volume_confirmation_stoploss and rvol is not None:
        rv = np.asarray(rvol, dtype=float)
        tightened = np.minimum(stop / np.sqrt(np.where(rv > 1.0, rv, 1.0)),
                               VOLUME_CONFIRMED_STOP_FLOOR)
        stop = np.where(rv > 1.0, tightened, stop)

    # atr_pct <= 0 means ATR was missing or zero; fall back to the static stop, as the
    # original did when `atr_pct_roll` was absent.
    return np.where(atr > 0, stop, params.static_stoploss)


def atr_stop_fraction(atr_pct: float, rvol: float, params: ExitParams) -> float:
    """The stop level relative to entry, as a negative fraction (scalar).

    THE extraction target -- the expression that was duplicated three ways:

        stop = -multiplier * atr_pct
        stop = max(min(stop, cap), floor)
        if volume confirmation and rvol > 1:
            stop = min(stop / sqrt(rvol), VOLUME_CONFIRMED_STOP_FLOOR)

    Delegates to `atr_stop_fraction_array` so there is one implementation, not two.

    The volume-confirmation step TIGHTENS the stop on unusually heavy volume (dividing a
    negative number by sqrt(rvol) > 1 moves it toward zero), which is the opposite of
    what "confirmation" might suggest.
    """
    return float(atr_stop_fraction_array(atr_pct, rvol, params))


def effective_stop_fraction(
    age_hours: float, atr_pct: float, rvol: float, params: ExitParams
) -> float:
    """`atr_stop_fraction`, except inside the grace window where the wide level applies.

    The grace window exists so entry noise does not stop a trade out immediately. Note
    it can be far wider than the normal stop (-0.27 at the production operating point
    against -0.02..-0.04), so a label horizon shorter than the window sees effectively
    no stop at all -- which is a real and easily-missed property of this policy.
    """
    if params.stoploss_grace_hours is None or params.stoploss_grace_level is None:
        raise ValueError(
            "stoploss_grace_hours and stoploss_grace_level must be supplied explicitly "
            "before the grace window can be evaluated. They have no default because this "
            "repository holds three different values (BaseStrategy.sell_params 2.6h/-0.17, "
            "NNNC_MLX_Open.json 6.5h/-0.27), and guessing would silently simulate an exit "
            "the strategy does not run."
        )
    if params.stoploss_grace_hours > 0.0 and age_hours < params.stoploss_grace_hours:
        return params.stoploss_grace_level
    return atr_stop_fraction(atr_pct, rvol, params)


def rolling_rvol(volume_series, window: int = 20, min_periods: int = 5) -> float:
    """Relative volume at the last bar, matching the live computation exactly.

    Extracted because the live path computed it inline, so an offline caller would
    otherwise have had to reproduce the window and min_periods -- the same kind of
    near-duplicate this module exists to prevent.
    """
    vol_mean = float(volume_series.rolling(window, min_periods=min_periods).mean().iloc[-1])
    cur_vol = float(volume_series.iloc[-1])
    if not vol_mean > 0:
        return 0.0
    return cur_vol / vol_mean


#: Why a bar can close a trade. `end_of_data` is distinguishable from a real exit so it
#: can be EXCLUDED rather than silently counted as a flat or losing outcome.
CLOSE_STOP = "stop"
CLOSE_TAKE_PROFIT = "take_profit"
CLOSE_HOLD_CAP = "hold_cap"
CLOSE_END_OF_DATA = "end_of_data"
#: FR-001's amendment.
CLOSE_METRIC_OVERBOUGHT = "metric_overbought"
CLOSE_UNCLOG_12H = "unclog_12h"
CLOSE_UNCLOG_1D = "unclog_1d"
CLOSE_ROI = "roi"
#: The LIVE `BaseStrategy.custom_exit`'s `cexit_max_days` branch returns the literal string
#: "max_hold" -- NOT `CLOSE_HOLD_CAP` ("hold_cap"), which is `simulate_forward`'s own,
#: differently-named reason for the SAME underlying rule (its bar-count loop bound running
#: out, feeding the offline table's `stop`/`take_profit`/`hold_cap`/`end_of_data` vocabulary,
#: data-model.md). The two call sites never compare these strings to each other, so nothing
#: requires them to match -- but `custom_exit_decision` is shared, and returning the wrong
#: one here would have silently relabelled every live max-hold exit from "max_hold" to
#: "hold_cap", which the custom_exit characterisation fixture exists to catch.
CLOSE_MAX_HOLD = "max_hold"


def custom_exit_decision(
    current_profit: float,
    age_hours: float,
    guard_metric: Optional[float],
    params: ExitParams,
) -> Optional[str]:
    """The single source of truth for `BaseStrategy.custom_exit`'s decision, MINUS the two
    things that stay in `BaseStrategy` as glue (this module's docstring, and
    contracts/exit-policy.md):

      * the three `rsi_*` branches -- INERT at the MINIMAL populator (no `rsi` column) and
        therefore never modelled here (FR-001's amendment documents rather than models them);
      * the `strategy_custom_exit` hook -- not overridden anywhere in the NNNC chain, so
        there is nothing for an offline simulator to call.

    Returns one of `CLOSE_METRIC_OVERBOUGHT` / `CLOSE_TAKE_PROFIT` / `CLOSE_UNCLOG_12H` /
    `CLOSE_UNCLOG_1D` / `CLOSE_MAX_HOLD`, or `None` to defer (live: to the native stop / ROI;
    offline: to `simulate_forward`'s own stop/ROI checks).

    `current_profit` and `age_hours` must both be computed from the bar's OPEN price and the
    trade's age at that same instant -- see this module's docstring for why (freqtrade calls
    `custom_exit` with `rate = row[OPEN_IDX]`). Passing a high- or low-based profit here would
    silently reproduce the exact bias this amendment exists to remove.
    """
    if params.cexit_enable_profit_checks and current_profit > 0.0:
        if guard_metric is not None and guard_metric > params.metric_overbought_threshold:
            return CLOSE_METRIC_OVERBOUGHT
        if current_profit > params.take_profit:
            return CLOSE_TAKE_PROFIT

    # Time-based exits -- UNCONDITIONAL on cexit_enable_profit_checks in the live code
    # (BaseStrategy.custom_exit: these sit after, not inside, the gated block above).
    if age_hours >= params.unclog_12h_hours and current_profit > params.unclog_12h_min_profit:
        return CLOSE_UNCLOG_12H
    if age_hours >= params.unclog_1d_hours and current_profit >= 0.0:
        return CLOSE_UNCLOG_1D
    if age_hours >= params.max_hold_days * 24.0:
        return CLOSE_MAX_HOLD
    return None


def simulate_forward(
    high, low, close, atr_pct, rvol, start_index, params: ExitParams,
    *, open_=None, guard_metric=None,
):
    """Replay the exit policy forward from a hypothetical long entry at `start_index`.

    Returns `(profit_fraction, hold_bars, close_reason)` where `profit_fraction` is NET of
    `fee_per_side` on BOTH sides (feature 021 FR-006a).

    `open_` and `guard_metric` are per-bar arrays, both OPTIONAL (feature 021's FR-001
    amendment). `None` means exactly what it means for `rvol` elsewhere in this module: "do
    not model this". Supplying neither reproduces this function's PRE-amendment behaviour
    bit-for-bit (`test_exit_policy_simulate_forward.py` / `test_exit_policy_grace.py` pin
    this). `research/outcomes/table.py` -- the real caller -- supplies both, because both
    columns already exist on every bar it builds from.

    PRECEDENCE, in the order below, per bar (this module's docstring has the full citation
    trail against freqtrade's source):

      1. IF `open_` is supplied: `custom_exit_decision` on the bar's OPEN price/age --
         `metric_overbought` / `take_profit` / `unclog_12h` / `unclog_1d`. (`max_hold` is
         deliberately NOT consulted here -- see below.)
      2. the native stop, on the bar's LOW (`effective_stop_fraction`, unchanged).
      3. IF `open_` is NOT supplied: the legacy high-based `take_profit` check (unchanged;
         this is the ONLY place this amendment preserves a pre-existing approximation rather
         than replacing it, and only for callers that cannot supply an open price).
      4. IF `params.roi` is supplied: the native ROI exit, on the bar's HIGH.

    SAME-BAR RULE PRECEDENCE WHEN `open_` IS NOT SUPPLIED, fixed and deliberate
    (contracts/exit-policy.md), UNCHANGED by this amendment: if a bar's low breaches the stop
    AND its high reaches the take-profit, the result is `stop`. OHLC alone cannot order
    intra-bar events, and the opposite choice would let an outcome table claim profits the
    live path would not have collected -- the phantom-return failure mode Constitution
    Principle V exists to prevent.

    WHY `max_hold` IS NOT CHECKED INSIDE THE LOOP. The loop's own bound (`last`, below) is
    already defined as exactly `age_hours >= params.max_hold_days * 24.0` -- the identical
    condition `custom_exit_decision`'s `CLOSE_MAX_HOLD` branch tests. Calling that branch from
    inside the loop would fire on the SAME bar the post-loop fallback already handles, but
    priced off that bar's OPEN instead of the tested, pinned `close` convention
    (`test_hold_cap_in_isolation`) -- two fill prices for one decision. The loop bound is the
    single source of truth for WHEN hold_cap fires; the post-loop fallback is the single
    source of truth for its fill price.

    The stop level is re-evaluated per bar via `effective_stop_fraction`, so the grace
    window is honoured. Note the grace level can be far wider than the normal stop, so a
    hold shorter than the window sees effectively no stop at all.
    """
    import numpy as np

    n = len(close)
    entry = float(close[start_index])
    if not entry > 0:
        return 0.0, 0, CLOSE_END_OF_DATA

    atr_entry = float(np.asarray(atr_pct)[start_index])
    rvol_entry = None if rvol is None else float(np.asarray(rvol)[start_index])

    bars_per_hour = 60.0 / params.timeframe_minutes
    max_bars = int(params.max_hold_days * 24 * bars_per_hour)
    tp_price = entry * (1.0 + params.take_profit)
    round_trip_fee = 2.0 * params.fee_per_side

    last = min(start_index + max_bars, n - 1)
    for k in range(start_index + 1, last + 1):
        age_hours = (k - start_index) / bars_per_hour

        # ---- 0. custom_exit's own decision, OPEN-based (feature 021 FR-001 amendment) ----
        if open_ is not None:
            profit_open = (float(open_[k]) / entry) - 1.0
            gm = None if guard_metric is None else float(guard_metric[k])
            decision = custom_exit_decision(profit_open, age_hours, gm, params)
            if decision in (
                CLOSE_METRIC_OVERBOUGHT, CLOSE_TAKE_PROFIT, CLOSE_UNCLOG_12H, CLOSE_UNCLOG_1D,
            ):
                return profit_open - round_trip_fee, k - start_index, decision

        # ---- 1. the native stop, on the LOW ------------------------------------------
        stop_frac = effective_stop_fraction(age_hours, atr_entry, rvol_entry, params)
        stop_price = entry * (1.0 + stop_frac)
        if float(low[k]) <= stop_price:
            return stop_frac - round_trip_fee, k - start_index, CLOSE_STOP

        # ---- 2. the legacy take-profit fallback, HIGH-based -- ONLY when the caller has
        #         not supplied an open price (pre-amendment behaviour, pinned by T009/T010)
        if open_ is None and float(high[k]) >= tp_price:
            return params.take_profit - round_trip_fee, k - start_index, CLOSE_TAKE_PROFIT

        # ---- 3. the native ROI exit, on the HIGH (feature 021 FR-001 amendment) ------
        if params.roi is not None:
            profit_high = (float(high[k]) / entry) - 1.0
            if profit_high >= params.roi:
                return params.roi - round_trip_fee, k - start_index, CLOSE_ROI

    if start_index + max_bars <= n - 1:                      # 4. hold cap
        exit_px = float(close[last])
        return (exit_px / entry) - 1.0 - round_trip_fee, last - start_index, CLOSE_HOLD_CAP

    return 0.0, last - start_index, CLOSE_END_OF_DATA        # 5. ran out of data
