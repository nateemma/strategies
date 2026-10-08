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


def simulate_forward(high, low, close, atr_pct, rvol, start_index, params: ExitParams):
    """Replay the exit policy forward from a hypothetical long entry at `start_index`.

    Returns `(profit_fraction, hold_bars, close_reason)` where `profit_fraction` is NET of
    `fee_per_side` on BOTH sides (feature 021 FR-006a).

    SAME-BAR RULE PRECEDENCE, fixed and deliberate (contracts/exit-policy.md): if a bar's
    low breaches the stop AND its high reaches the take-profit, the result is `stop`. OHLC
    alone cannot order intra-bar events, and the opposite choice would let an outcome table
    claim profits the live path would not have collected -- the phantom-return failure mode
    Constitution Principle V exists to prevent. The cost of being wrong here is
    asymmetric: a pessimistic tie understates a trigger, an optimistic one invents edge.

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
        stop_frac = effective_stop_fraction(age_hours, atr_entry, rvol_entry, params)
        stop_price = entry * (1.0 + stop_frac)

        if float(low[k]) <= stop_price:                      # 1. stop WINS a same-bar tie
            return stop_frac - round_trip_fee, k - start_index, CLOSE_STOP
        if float(high[k]) >= tp_price:                       # 2. take-profit
            return params.take_profit - round_trip_fee, k - start_index, CLOSE_TAKE_PROFIT

    if start_index + max_bars <= n - 1:                      # 3. hold cap
        exit_px = float(close[last])
        return (exit_px / entry) - 1.0 - round_trip_fee, last - start_index, CLOSE_HOLD_CAP

    return 0.0, last - start_index, CLOSE_END_OF_DATA        # 4. ran out of data
