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

AMENDED 2026-10-08 (fee double-count fix). The reconciliation table's realised
`profit_ratio` (itself NET of fees -- freqtrade's `Trade.calc_profit_ratio` docstring:
"Calculates the profit as ratio (including fee)"; `calculate_profit`: "All calculations
include fees") landed almost exactly ON the ROI threshold (+0.050008 vs 0.0500) and BELOW
the stop threshold by one fee (-0.271446 vs -0.2700). That is because freqtrade's
`should_exit` (`freqtrade/strategy/interface.py`) computes
`current_profit = trade.calc_profit_ratio(current_rate)` -- NET -- and passes it straight
into `custom_exit` (feeding this module's `custom_exit_decision`); separately
`current_profit_best = trade.calc_profit_ratio(current_rate_best)` (HIGH-based, also NET)
is what `min_roi_reached` compares against `self.minimal_roi` (`current_profit > roi`,
interface.py's `min_roi_reached`). Both `cexit_take_profit`/`unclog_12h`/`unclog_1d`
thresholds and the native `minimal_roi` level are therefore NET targets by construction:
there is no separate fee subtraction anywhere on freqtrade's side for these checks. The
native stop is different -- it is a PRICE level (`current_rate <= stop_price`), with no
fee netting in the comparison at all, so a fee properly applies on top of it once a trade
actually fills there.

Pre-fix, `simulate_forward` got both halves wrong for the custom-exit/ROI group: it
compared a GROSS, not-yet-fee-adjusted profit against these NET thresholds (over-reporting
how far a bar was from the fee-adjusted target), and then subtracted `round_trip_fee` a
SECOND time from the returned value. The fix (below) converts to net ONCE, before the
comparison, for exactly these checks, and returns that same net figure with no further
subtraction -- matching the live contract, where `custom_exit_decision`'s `current_profit`
argument has always been documented (and, live, always been) a NET figure. The stop,
`hold_cap` and `end_of_data` branches are UNCHANGED: they close at a traded PRICE, so the
fee is realised on top, exactly once, same as before.

AMENDED 2026-10-08 (entry-price convention fix). `simulate_forward` entered at
`close[start_index]`; the live strategy enters at the OPEN of the bar whose timestamp is
`trade.open_date` -- the SAME bar, opposite end. Reconciliation against 1,410 matched
control-arm trades found `divergence = table_profit - arm_profit` correlated -0.495 with
`entry_bar_move = close[t]/open[t] - 1`, with a monotonic quintile pattern (fell-most
+0.0160 to rose-most -0.0190) and a regression slope of -1.072 (intercept ~0) -- a pure
entry-price convention mismatch predicts exactly -1.0, so this was conclusive on the
mechanism, not a coincidental correlation.

The fix, gated on `open_` being supplied (same flag that already gates every other
open-based branch in this function -- see FR-001's amendment above): `entry =
open_[start_index]`, not `close[start_index]`. Because entry now happens at the bar's open,
that SAME bar's own high/low are reachable immediately after entry, so the replay loop
starts at `start_index` (age 0 hours there) instead of `start_index + 1`.

`open_=None` DELIBERATELY KEEPS the function's pre-fix convention bit-for-bit: entry at
`close[start_index]`, replay starting at `start_index + 1`, the legacy HIGH-based
`take_profit` fallback, the no-gap-detection stop fill. This is the documented-non-faithful
option (as opposed to raising) -- chosen because the alternative, forcing `open_` to be
mandatory, would delete the ONLY code path `test_exit_policy_simulate_forward.py` pins:
once `open_` is supplied, `take_profit` is resolved exclusively via the OPEN-based
`custom_exit_decision` check, which this module's docstring already establishes fires
BEFORE the native stop -- i.e. the HIGH-based "same-bar tie, stop wins" rule that file pins
has no code path left to exercise once `open_` is mandatory. Making `open_=None` raise
would therefore not just need new test SETUP, it would retire that file's entire reason
to exist. Every REAL caller (`research/outcomes/table.py`) already always supplies
`open_`, so this fallback is reachable only from the handful of pre-FR-001-amendment tests
that predate `open_` support; it is not a trap a new caller can stumble into unknowingly,
because every other branch this flag already gates (the amendment's own module docstring,
above) has carried the exact same "`None` means do not model this, clearly documented"
contract since that amendment landed.

Expected effect, for callers that DO supply `open_`: the divergence/entry-bar-move
correlation and regression slope should collapse toward zero, and the subset biases (stop
exits optimistic, profit exits pessimistic) should shrink by roughly 25-30% -- NOT to
zero, since this mismatch is only part of a larger, still-undiagnosed residual.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Bump whenever `simulate_forward`'s or `custom_exit_decision`'s BEHAVIOUR changes --
#: i.e. the same inputs now produce a different `profit_fraction`/`close_reason`, not just
#: a different *parameter value* (parameter changes are already covered by `ExitConfig`'s
#: hash over `ExitParams`' fields). `research/outcomes/table.py`'s `ExitConfig.config_hash`
#: folds this constant into its hash specifically so a behavioural change that is pure CODE
#: (no `ExitParams` field changed) still invalidates the cache. Forgetting to bump this is a
#: silent-staleness bug of exactly the kind FR-007 (specs/021-label-exit-coordination) exists
#: to prevent: run 3's fill-price fix landed with the SAME `config_hash` as run 2 because
#: nothing forced the hash to notice the code had changed underneath it.
#:
#: Introduced at 1 (the implicit, never-versioned behaviour every prior run silently
#: depended on), bumped to 2 for the fee double-count fix, and bumped to 3 for the
#: entry-price convention fix (module docstring, "AMENDED 2026-10-08 (entry-price
#: convention fix)"): when `open_` is supplied, `simulate_forward` now enters at
#: `open_[start_index]` instead of `close[start_index]` and replays from `start_index`
#: instead of `start_index + 1`, so the same inputs can produce a different
#: `profit_fraction`/`close_reason`/`hold_bars` than version 2 did. (Callers that omit
#: `open_` are unaffected -- see the module docstring for why that path is deliberately
#: left as the pre-fix convention.) Every version bump happens in the same commit as the
#: behaviour change it covers, so the version this module ships with is never itself stale.
SIMULATION_LOGIC_VERSION = 3

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

    `current_profit` MUST be NET of round-trip fees, matching the live contract exactly:
    `should_exit` (interface.py) computes it via `trade.calc_profit_ratio`, which is already
    fee-inclusive ("Calculates the profit as ratio (including fee)"), and passes that straight
    into `custom_exit` with no further adjustment. This function performs NO fee arithmetic
    itself -- an offline caller (`simulate_forward`) that has only a raw price ratio must net
    the fee out BEFORE calling this, not after reading the decision back.
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
    `fee_per_side` on BOTH sides (feature 021 FR-006a) -- HOW it gets there differs by exit
    type, fixed 2026-10-08 (see this module's docstring, "AMENDED ... fee double-count
    fix"):

      * `stop` / `hold_cap` / `end_of_data` close at a traded PRICE (the stop level, or the
        bar's open/close); `round_trip_fee` is subtracted from that price-based fraction
        exactly once, same as always.
      * `metric_overbought` / `take_profit` / `unclog_12h` / `unclog_1d` (the `open_`-based
        `custom_exit_decision` branches) and `roi` compare and RETURN a NET figure --
        `(price/entry - 1) - round_trip_fee` -- because freqtrade's own thresholds for these
        (`cexit_take_profit`, `unclog_12h_min_profit`, `unclog_1d`'s zero-profit floor, and
        `minimal_roi`) are themselves compared against a NET `current_profit`
        (`trade.calc_profit_ratio`, fee-inclusive by construction). Computing a GROSS profit,
        comparing it against these NET thresholds, and then subtracting the fee a second time
        from the return value double-counts the fee and is WRONG at both ends; this function
        nets the fee out ONCE, before the comparison.

    `open_` and `guard_metric` are per-bar arrays, both OPTIONAL (feature 021's FR-001
    amendment). `None` means exactly what it means for `rvol` elsewhere in this module: "do
    not model this". Supplying neither reproduces this function's PRE-amendment behaviour
    bit-for-bit (`test_exit_policy_simulate_forward.py` / `test_exit_policy_grace.py` pin
    this). `research/outcomes/table.py` -- the real caller -- supplies both, because both
    columns already exist on every bar it builds from.

    ENTRY PRICE, gated on the SAME `open_ is not None` flag (entry-price convention fix,
    module docstring, "AMENDED 2026-10-08 (entry-price convention fix)"): `open_[start_index]`
    when supplied, `close[start_index]` -- the pre-fix, non-faithful convention -- when not.
    Because entry then happens at the bar's open, that SAME bar's own high/low are reachable
    immediately after entry, so the replay loop starts AT `start_index` (age 0 hours) rather
    than `start_index + 1` -- again, only when `open_` is supplied; the legacy path still
    enters at `close` and so still starts replaying the bar after.

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

    STOP FILL PRICE ("the grace cliff" fix, T021 reconciliation follow-up). A stop exit's
    realised fraction is NOT always `stop_frac` -- that assumes a fill AT the stop level,
    which freqtrade's `Backtesting._get_close_rate_for_stoploss` only does when the stop
    price is within the bar's range. When the grace window expires and the stop snaps in
    from `stoploss_grace_level` to the normal ATR level, a bar can open already through the
    (tighter) stop; freqtrade fills that at the bar's OPEN, not the stop. IF `open_` is
    supplied, this function matches that: `open[k] >= stop_price` fills at `stop_frac`
    (unchanged); otherwise it fills at `open[k]/entry - 1.0` (worse). IF `open_` is NOT
    supplied there is no per-bar signal to detect the gap, so the fill stays `stop_frac` --
    optimistic, and only acceptable for pre-amendment callers that cannot provide `open_`.
    """
    import numpy as np

    n = len(close)
    # ENTRY PRICE (entry-price convention fix): open-based once `open_` is supplied,
    # close-based (the pre-fix, non-faithful convention) otherwise -- see this function's
    # docstring for why the fallback stays close-based rather than raising.
    entry = float(open_[start_index]) if open_ is not None else float(close[start_index])
    if not entry > 0:
        return 0.0, 0, CLOSE_END_OF_DATA

    atr_entry = float(np.asarray(atr_pct)[start_index])
    rvol_entry = None if rvol is None else float(np.asarray(rvol)[start_index])

    bars_per_hour = 60.0 / params.timeframe_minutes
    max_bars = int(params.max_hold_days * 24 * bars_per_hour)
    tp_price = entry * (1.0 + params.take_profit)
    round_trip_fee = 2.0 * params.fee_per_side

    last = min(start_index + max_bars, n - 1)
    # LOOP START (entry-price convention fix): once entry happens at the bar's OPEN, that
    # same bar's own high/low are reachable immediately after entry, so the replay must
    # start AT start_index (age 0 hours), not start_index + 1 -- but only when `open_` is
    # supplied; the legacy (no `open_`) path still enters at `close`, so it still starts
    # replaying the bar after, exactly as before this fix.
    loop_start = start_index if open_ is not None else start_index + 1
    for k in range(loop_start, last + 1):
        age_hours = (k - start_index) / bars_per_hour

        # ---- 0. custom_exit's own decision, OPEN-based (feature 021 FR-001 amendment) ----
        # NET, not gross (fee double-count fix, 2026-10-08): custom_exit_decision's
        # thresholds (cexit_take_profit / unclog_12h_min_profit / unclog_1d's zero floor)
        # are themselves NET targets live (see custom_exit_decision's and this function's
        # docstrings), so the comparison -- not just the return value -- must be net too.
        # At k == start_index (when `open_` is supplied, so this is reachable) this is
        # always a no-op: open_[start_index] == entry by definition, so profit_open_net ==
        # -round_trip_fee <= 0, and age_hours == 0 fails every time-based threshold too.
        if open_ is not None:
            profit_open_net = (float(open_[k]) / entry) - 1.0 - round_trip_fee
            gm = None if guard_metric is None else float(guard_metric[k])
            decision = custom_exit_decision(profit_open_net, age_hours, gm, params)
            if decision in (
                CLOSE_METRIC_OVERBOUGHT, CLOSE_TAKE_PROFIT, CLOSE_UNCLOG_12H, CLOSE_UNCLOG_1D,
            ):
                return profit_open_net, k - start_index, decision

        # ---- 1. the native stop, on the LOW ------------------------------------------
        stop_frac = effective_stop_fraction(age_hours, atr_entry, rvol_entry, params)
        stop_price = entry * (1.0 + stop_frac)
        if float(low[k]) <= stop_price:
            # FILL PRICE, not just the decision (T021 reconciliation follow-up, the
            # "grace cliff" fix). freqtrade's own
            # `Backtesting._get_close_rate_for_stoploss` (optimize/backtesting.py) does
            # NOT always fill a stop exit at the stop level: for a long,
            #   if stoploss_value > row[HIGH_IDX]: return row[OPEN_IDX]   # gapped past it
            #   ...
            #   return stoploss_value                                     # normal case
            # i.e. when the whole candle is already below the stop (a gap), the fill is
            # the candle's OPEN, not the stop price -- the stop price was never available
            # to trade at. During `stoploss_grace_level`'s wide window a trade can drift
            # far underwater; when grace expires and the stop snaps back to the normal
            # ATR level, `low[k] <= stop_price` can already be true at the bar's OPEN,
            # not just intra-bar. Modelling every such exit as a fill AT the stop
            # (the pre-fix code, unconditionally `stop_frac - round_trip_fee`) is the
            # optimistic direction Constitution Principle V exists to catch: real stop
            # exits were found up to -0.2714 (== stoploss_grace_level) against a
            # stop-clamped floor of about -0.042 net of fees.
            #
            # `open[k] >= stop_price` is this function's bar-local proxy for "no gap":
            # freqtrade's rule, verbatim from Backtesting._get_close_rate_for_stoploss
            # (optimize/backtesting.py): for a long it fills at the OPEN only when
            # `stoploss_value > row[HIGH_IDX]` -- the WHOLE candle gapped past the stop,
            # so the stop price was never tradeable. Otherwise the stop sits within the
            # candle's range and fills AT the stop.
            #
            # Keyed on HIGH, not on open. An earlier version of this branch used
            # `open[k] < stop_price`, which differs on the narrow case
            # `open <= stop <= high` -- a candle that opens through the stop but trades
            # back above it. freqtrade fills that at the stop; the open-based rule filled
            # it at the open. That error was PESSIMISTIC rather than optimistic, so it
            # would not have inflated a trigger's value -- but it would have understated
            # one, and fidelity to the live policy is the point, not conservatism.
            if open_ is not None:
                if stop_price > float(high[k]):
                    realised = (float(open_[k]) / entry) - 1.0
                else:
                    realised = stop_frac
            else:
                # No open price supplied: this is the pre-amendment fallback, and it is
                # deliberately optimistic (fill AT the stop), because without `open_`
                # there is no per-bar signal to detect a gap at all. Callers that care
                # about the gap-down fill (every real outcome-table builder) must supply
                # `open_`; this branch exists only so pre-amendment callers
                # (`test_exit_policy_simulate_forward.py` / `test_exit_policy_grace.py`)
                # keep their bit-for-bit pinned behaviour.
                realised = stop_frac
            return realised - round_trip_fee, k - start_index, CLOSE_STOP

        # ---- 2. the legacy take-profit fallback, HIGH-based -- ONLY when the caller has
        #         not supplied an open price (pre-amendment behaviour, pinned by T009/T010)
        if open_ is None and float(high[k]) >= tp_price:
            return params.take_profit - round_trip_fee, k - start_index, CLOSE_TAKE_PROFIT

        # ---- 3. the native ROI exit, on the HIGH (feature 021 FR-001 amendment) ------
        # NET, not gross (fee double-count fix, 2026-10-08): `minimal_roi` is compared
        # live against `trade.calc_profit_ratio` (NET, interface.py's `min_roi_reached`),
        # so this must compare and return the same NET figure, not the fixed threshold
        # minus a fee.
        if params.roi is not None:
            profit_high_net = (float(high[k]) / entry) - 1.0 - round_trip_fee
            if profit_high_net >= params.roi:
                return profit_high_net, k - start_index, CLOSE_ROI

    if start_index + max_bars <= n - 1:                      # 4. hold cap
        exit_px = float(close[last])
        return (exit_px / entry) - 1.0 - round_trip_fee, last - start_index, CLOSE_HOLD_CAP

    return 0.0, last - start_index, CLOSE_END_OF_DATA        # 5. ran out of data
