"""The remaining 10 distinct backward factors from T025's audit (FR-012).

WHY THIS FILE EXISTS. T025 found 19 `LabelMethod` members reduce to a trigger, and those 19
carry **14 distinct** backward factors. The first pass registered 4 (`adx`, `breakout`,
`breakout_gbb`, `gbb`) and justified stopping there as "not one per labeler, which would pad
the leaderboard with duplicate rows". That justification covers only 5 members — the ones
whose factor really is identical to a registered one (`future_sortino` and
`future_expectancy` share `adx`; `breakout_tb` shares `breakout`; `gbb_path` and
`gbb_fwdmae` share `gbb`). The other 10 are genuinely different conditions and were simply
missing. This file closes that gap so the leaderboard covers the audit rather than a subset
of it.

Every condition is read off the live `Framework/TrainingSignals.py`, with the labeler's own
default thresholds, and each port keeps the labeler's NaN convention rather than tidying it
— the same discipline as `builtin_triggers.py`, for the same reason: a tidied NaN rule
silently diverges from the label it claims to represent.

TWO OF THE TEN ARE UNSCOREABLE, which is itself a finding. `trends` needs
`flow`/`regime`/`risk`/`momentum` and `bands` needs eight band columns; the populator the
outcome table is built from (`DataframePopulator`'s MINIMAL set, via
`research.clfstudy.data.populate`) supplies NONE of the first four and only two of the eight.
Both therefore declare their real requirements and the registry REFUSES to evaluate them,
rather than scoring a degraded version under the labeler's name. That is a third category
FR-012 did not anticipate: a member that reduces to a genuine backward factor which is not
computable on the available feature set. `trends` is the dangerous one — its NaN fill of 0.0
reads as DECREASE/BEAR/NEGATIVE and VOTES YES, so a silent evaluation would have fired on
almost every bar and looked like a wildly permissive trigger rather than a missing input.

ONE GENUINE JUDGEMENT, recorded rather than hidden. The five vote-family factors
(`technical_indicators`, `indicators2`, `indicators3`, `indicators4`,
`breakout_consensus`) are all "at least N of these indicator conditions hold", but over
DIFFERENT indicator sets with different thresholds and different N. They are therefore
distinct conditions, not one condition five times, and each gets a row. They are the
fiddliest ports here and the most likely to be judged redundant later; the measurement is
what settles that, not an opinion in advance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.outcomes.triggers import Trigger, register

# -------------------------------------------------------------------------------------------------
# helpers, mirroring the labelers' own column access and NaN handling
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Trigger families (feature 024 FR-001) -- what each condition TESTS, not what it is called
# -------------------------------------------------------------------------------------------------

_FAMILY = {
    "breakout_vol": ("momentum", "a new 20-bar high on relative volume > 1.5: strength"),
    "breakout_squeeze": ("momentum", "a new 20-bar high out of a volatility squeeze: strength"),
    "breakout_bb": (
        "momentum",
        "bb_position >= 0.6 with a wide band: buys HIGH band position (not a breakout)",
    ),
    "votes_tech": ("reversion", "4 of 6 OVERSOLD conditions: buys weakness"),
    "votes_ind2": ("unassigned", "4-of-7 vote, ~10 trades/day: broad filter, no direction"),
    "votes_ind3": ("unassigned", "3-of-4 vote, ~10 trades/day: broad filter, no direction"),
    "votes_ind4": ("unassigned", "3-of-5 vote, ~12 trades/day: broad filter, no direction"),
    "votes_consensus": ("unassigned", "votes_tech with 3 comparisons flipped: no direction"),
    "trends": ("unassigned", "UNSCOREABLE (columns absent): cannot be classified"),
    "bands": ("unassigned", "UNSCOREABLE (columns absent): cannot be classified"),
    "breakout_gm": ("momentum", "a new 20-bar high AND guard_metric > 0.2: strength"),
    "breakout_vol_gm": ("momentum", "breakout_vol AND guard_metric > 0.2: strength"),
    "breakout_squeeze_gm": ("momentum", "breakout_squeeze AND guard_metric > 0.2: strength"),
    "breakout_gbb_nofresh": (
        "momentum",
        "breakout_gbb minus fresh/held breaks: still buys high band position",
    ),
    "failed_breakout": ("reversion", "price back below a broken 20-bar high: fades a failed move"),
}


def _fam(name: str) -> dict:
    family, reason = _FAMILY[name]
    return {"family": family, "family_reason": reason}


def _col(df: pd.DataFrame, name: str, *, fill: float = 0.0) -> np.ndarray:
    """`df.get(name, zeros)` then `nan_to_num`, exactly as the labelers do it.

    `fill` follows the specific labeler: most use 0.0, `labels_breakout_vol` uses 1.0 for
    `rvol` (so a missing rvol is NEUTRAL there rather than failing its `> 1.5` test).
    """
    n = len(df)
    s = df.get(name, pd.Series(np.full(n, fill), index=df.index))
    return np.nan_to_num(np.asarray(s, dtype=float), nan=fill)


def _prior_high_breakout(df: pd.DataFrame, lookback: int) -> np.ndarray:
    """`labels_breakout`'s condition, shared by the two variants below.

    Imported rather than re-derived would be better, but `builtin_triggers._breakout` folds
    in its own warm-up fill; this returns the raw mask plus the NaN positions so each
    variant can conjoin before filling, which is what the labelers do.
    """
    high = pd.Series(np.asarray(df["high"], dtype=float))
    prior = high.rolling(lookback).max().shift(1).to_numpy()
    close = np.asarray(df["close"], dtype=float)
    out = close > prior
    return np.where(np.isnan(prior), False, out)


# -------------------------------------------------------------------------------------------------
# breakout variants
# -------------------------------------------------------------------------------------------------

BREAKOUT_LOOKBACK = 20          # labels_breakout_vol / _squeeze default
BREAKOUT_VOL_RVOL_MIN = 1.5     # labels_breakout_vol default
BREAKOUT_SQUEEZE_PCT = 0.3      # labels_breakout_squeeze default
BREAKOUT_BB_POSITION_THRESHOLD = 0.6    # labels_breakout_bb defaults
BREAKOUT_BB_WIDTH_THRESHOLD = 0.035


def _breakout_vol(df: pd.DataFrame) -> pd.Series:
    """`labels_breakout_vol`: breakout confirmed by relative volume.

    Note the fill: a missing `rvol` becomes 1.0 here, not 0.0, so it FAILS `> 1.5` without
    being treated as zero volume. That differs from `gbb`'s convention and is the labeler's
    own choice.
    """
    mask = _breakout_vol_mask(df)
    return pd.Series(mask, index=df.index)


def _breakout_vol_mask(df: pd.DataFrame) -> np.ndarray:
    return _prior_high_breakout(df, BREAKOUT_LOOKBACK) & (
        _col(df, "rvol", fill=1.0) > BREAKOUT_VOL_RVOL_MIN
    )


def _breakout_squeeze(df: pd.DataFrame) -> pd.Series:
    """`labels_breakout_squeeze`: breakout out of a volatility squeeze.

    `squeeze` is `bb_width.shift(1) <= bb_width.rolling(100).quantile(0.3)` — the PREVIOUS
    bar's width against a trailing quantile, so it is causal (both terms are closed bars).
    The rolling quantile's 100-bar warm-up is NaN and filled False, which is a real
    non-firing rather than a guess: before 100 bars there is no quantile to compare against.
    """
    bb_width = pd.Series(_col(df, "bb_width"))
    bw_prev = bb_width.shift(1)
    bw_thresh = bb_width.rolling(100).quantile(BREAKOUT_SQUEEZE_PCT)
    squeeze = (bw_prev <= bw_thresh).fillna(False).to_numpy()
    return pd.Series(_prior_high_breakout(df, BREAKOUT_LOOKBACK) & squeeze, index=df.index)


def _breakout_bb(df: pd.DataFrame) -> pd.Series:
    """`labels_breakout_bb`: position high in the Bollinger band AND the band wide.

    Not a breakout at all despite the name — it reads `bb_position`/`bb_width` and never
    computes a prior high. Kept under the labeler's name so the audit table maps 1:1.
    """
    mask = (_col(df, "bb_position") >= BREAKOUT_BB_POSITION_THRESHOLD) & (
        _col(df, "bb_width") > BREAKOUT_BB_WIDTH_THRESHOLD
    )
    return pd.Series(mask, index=df.index)


# -------------------------------------------------------------------------------------------------
# the vote family — five DIFFERENT indicator sets, hence five rows
# -------------------------------------------------------------------------------------------------

def _votes(df: pd.DataFrame, conditions: list[np.ndarray], min_votes: int) -> pd.Series:
    stacked = np.sum([c.astype(int) for c in conditions], axis=0)
    return pd.Series(stacked >= min_votes, index=df.index)


TECH_MIN_VOTES = 4          # labels_technical_indicators default
IND2_MIN_VOTES = 4          # labels_indicators2 default
IND3_MIN_VOTES = 3          # labels_indicators3 default
IND4_MIN_VOTES = 3          # labels_indicators4 default
CONSENSUS_MIN_VOTES = 4     # labels_breakout_consensus default


def _votes_tech(df: pd.DataFrame) -> pd.Series:
    """`labels_technical_indicators`: 4 of 6 oversold conditions."""
    return _votes(df, [
        _col(df, "adx_scaled") >= 0.1,
        _col(df, "aroonosc_scaled") <= -0.2,
        _col(df, "guard_metric") <= -0.2,
        _col(df, "sar_ratio") >= 0.2,
        _col(df, "bb_width") >= 0.017,
        _col(df, "vwap_ratio") >= 0.2,
    ], TECH_MIN_VOTES)


def _votes_ind2(df: pd.DataFrame) -> pd.Series:
    """`labels_indicators2`: 4 of 7. Note `bb_width <= 0.011` — this set wants a NARROW
    band where `technical_indicators` wants a wide one, so the two are not variants."""
    return _votes(df, [
        _col(df, "atr_norm") >= 0.5,
        _col(df, "gain_norm") <= -0.5,
        _col(df, "rsi_scaled") <= -0.0,
        _col(df, "mfi_scaled") <= -0.0,
        _col(df, "bb_width") <= 0.011,
        _col(df, "log_volume_norm") >= 0.2,
        _col(df, "close_norm") <= -0.2,
    ], IND2_MIN_VOTES)


def _votes_ind3(df: pd.DataFrame) -> pd.Series:
    """`labels_indicators3`: 3 of 4."""
    return _votes(df, [
        _col(df, "close_norm") < -0.0,
        _col(df, "adx_scaled") > -0.2,
        _col(df, "trend_mode") > 0,
        _col(df, "guard_metric") < -0.0,
    ], IND3_MIN_VOTES)


def _votes_ind4(df: pd.DataFrame) -> pd.Series:
    """`labels_indicators4`: 3 of 5."""
    return _votes(df, [
        _col(df, "rsi_scaled") < 0.0,
        _col(df, "mfi_scaled") < 0.0,
        _col(df, "ema_fast_norm") < 0.0,
        _col(df, "fastk_scaled") < 0.0,
        _col(df, "di_diff_scaled") > 0.0,
    ], IND4_MIN_VOTES)


def _votes_consensus(df: pd.DataFrame) -> pd.Series:
    """`labels_breakout_consensus`: 4 of 6, the SAME six columns as
    `technical_indicators` with three comparisons FLIPPED (aroonosc and guard_metric now
    want high values, not low). The clearest case in the family for why these are distinct
    conditions and not one condition repeated: same inputs, opposite regime."""
    return _votes(df, [
        _col(df, "adx_scaled") >= 0.1,
        _col(df, "aroonosc_scaled") >= 0.2,
        _col(df, "guard_metric") >= 0.2,
        _col(df, "sar_ratio") >= 0.2,
        _col(df, "bb_width") >= 0.017,
        _col(df, "vwap_ratio") >= 0.2,
    ], CONSENSUS_MIN_VOTES)


# -------------------------------------------------------------------------------------------------
# trends and bands
# -------------------------------------------------------------------------------------------------

TRENDS_MIN_VOTES = 2        # labels_trends: "Require 2 out of 4 conditions"


def _trends(df: pd.DataFrame) -> pd.Series:
    """`labels_trends`'s `buy_trend_mask`: 2 of 4 regime-state conditions.

    These are CATEGORICAL columns (`flow`/`regime`/`momentum` are 0/1/2 enums, `risk` is
    0/1/2), compared by equality rather than threshold, so the NaN fill of 0.0 is not
    neutral here — it reads as DECREASE/BEAR/NEGATIVE and VOTES YES. Reproduced because
    that is the live behaviour, flagged because it is the opposite of harmless.
    """
    return _votes(df, [
        _col(df, "flow") == 0,
        _col(df, "regime") == 0,
        _col(df, "risk") <= 1,
        _col(df, "momentum") == 0,
    ], TRENDS_MIN_VOTES)


_BAND_PAIRS = (
    ("bb_lowerband", "bb_upperband"),
    ("dc_lowerband", "dc_upperband"),
    ("kc_lowerband", "kc_upperband"),
    ("lw_lowerband", "lw_upperband"),
)


def _bands(df: pd.DataFrame) -> pd.Series:
    """`labels_bands`: close crossing ABOVE any of four lower or upper bands.

    `cross_above(curr, band)` is `(curr > band) & (prev_close <= prev_band)`, where the
    labeler takes `prev` with `np.roll(..., 1)`. `np.roll` WRAPS, so the first bar compares
    against the LAST bar of the series — a real wrap-around artefact in the live labeler.
    Reproduced for fidelity but the first bar is then forced False, because a wrapped
    comparison is not a signal and a single bar is not worth preserving a bug for. Noted
    rather than silently matched: this is the one place this port knowingly differs from the
    labeler, and it differs on exactly one bar per pair.
    """
    close = np.asarray(df["close"], dtype=float)
    prev_close = np.roll(close, 1)
    crossed = np.zeros(len(df), dtype=bool)
    for lower, upper in _BAND_PAIRS:
        for name in (lower, upper):
            if name not in df.columns:
                continue
            band = np.asarray(df[name], dtype=float)
            prev_band = np.roll(band, 1)
            hit = (close > band) & (prev_close <= prev_band)
            crossed |= np.where(np.isnan(band) | np.isnan(prev_band), False, hit)
    if len(crossed):
        crossed[0] = False          # kill the np.roll wrap-around
    return pd.Series(crossed, index=df.index)


# -------------------------------------------------------------------------------------------------

def register_remaining_audit_triggers(*, replace: bool = False) -> list[Trigger]:
    """The 10 distinct factors the first pass missed (FR-012)."""
    specs = [
        ("breakout_vol", _breakout_vol, ("high", "close", "rvol"), "labels_breakout_vol",
         f"breakout AND rvol > {BREAKOUT_VOL_RVOL_MIN} (missing rvol fills to 1.0, not 0.0)"),
        ("breakout_squeeze", _breakout_squeeze, ("high", "close", "bb_width"),
         "labels_breakout_squeeze",
         f"breakout AND prior bb_width <= its trailing {BREAKOUT_SQUEEZE_PCT:.0%} quantile(100)"),
        ("breakout_bb", _breakout_bb, ("bb_position", "bb_width"), "labels_breakout_bb",
         f"bb_position >= {BREAKOUT_BB_POSITION_THRESHOLD} AND bb_width > "
         f"{BREAKOUT_BB_WIDTH_THRESHOLD} — NOT a breakout despite the name"),
        ("votes_tech", _votes_tech,
         ("adx_scaled", "aroonosc_scaled", "guard_metric", "sar_ratio", "bb_width",
          "vwap_ratio"), "labels_technical_indicators",
         f"{TECH_MIN_VOTES} of 6 oversold conditions"),
        ("votes_ind2", _votes_ind2,
         ("atr_norm", "gain_norm", "rsi_scaled", "mfi_scaled", "bb_width",
          "log_volume_norm", "close_norm"), "labels_indicators2",
         f"{IND2_MIN_VOTES} of 7; wants a NARROW band where votes_tech wants a wide one"),
        ("votes_ind3", _votes_ind3,
         ("close_norm", "adx_scaled", "trend_mode", "guard_metric"), "labels_indicators3",
         f"{IND3_MIN_VOTES} of 4"),
        ("votes_ind4", _votes_ind4,
         ("rsi_scaled", "mfi_scaled", "ema_fast_norm", "fastk_scaled", "di_diff_scaled"),
         "labels_indicators4", f"{IND4_MIN_VOTES} of 5"),
        ("votes_consensus", _votes_consensus,
         ("adx_scaled", "aroonosc_scaled", "guard_metric", "sar_ratio", "bb_width",
          "vwap_ratio"), "labels_breakout_consensus",
         f"{CONSENSUS_MIN_VOTES} of 6 — same columns as votes_tech, three comparisons FLIPPED"),
        ("trends", _trends, ("flow", "regime", "risk", "momentum"), "labels_trends",
         f"{TRENDS_MIN_VOTES} of 4 regime states — UNSCOREABLE: all four columns absent "
         f"from the populator (NaN would fill to 0.0, which VOTES YES, so a silent "
         f"evaluation would fire on ~everything)"),
        # Declares ALL EIGHT band columns, not just the two that exist. The populator the
        # outcome table uses supplies only `bb_lowerband`/`bb_upperband`, so this trigger
        # REFUSES to evaluate -- which is correct. Declaring only `close` would have let it
        # silently score a Bollinger-only subset under the labeler's name, i.e. a different
        # condition wearing the same label. The registry's required-columns guard is doing
        # exactly its job here; see UNSCOREABLE below.
        ("bands", _bands,
         ("close", "bb_lowerband", "bb_upperband", "dc_lowerband", "dc_upperband",
          "kc_lowerband", "kc_upperband", "lw_lowerband", "lw_upperband"),
         "labels_bands",
         "close crossing above any of 4 lower/upper bands — UNSCOREABLE: 6 of 8 band "
         "columns absent from the populator"),
    ]
    return [
        register(Trigger(name=n, evaluate=f, required_columns=c, source=s, description=d,
                         **_fam(n)), replace=replace)
        for n, f, c, s, d in specs
    ]


# -------------------------------------------------------------------------------------------------
# Momentum-guarded breakouts (feature 023, Addendum 4, 2026-10-11)
# -------------------------------------------------------------------------------------------------

#: `breakout_gbb`'s guard direction. `breakout_gbb` trades on this condition and the true price
#: breakouts do not, so these test whether that guard is what the breakouts are missing.
MOMENTUM_GUARD_MIN = 0.2


def _momentum_guard(df: pd.DataFrame) -> np.ndarray:
    # NaN fills to 0.0, which FAILS `> 0.2` -- the same convention as `breakout_gbb`.
    return _col(df, "guard_metric") > MOMENTUM_GUARD_MIN


def _breakout_gm(df: pd.DataFrame) -> pd.Series:
    return pd.Series(_prior_high_breakout(df, BREAKOUT_LOOKBACK) & _momentum_guard(df),
                     index=df.index)


def _breakout_vol_gm(df: pd.DataFrame) -> pd.Series:
    return pd.Series(_breakout_vol_mask(df) & _momentum_guard(df), index=df.index)


def _breakout_squeeze_gm(df: pd.DataFrame) -> pd.Series:
    return pd.Series(_breakout_squeeze(df).to_numpy() & _momentum_guard(df), index=df.index)


def register_momentum_guard_triggers(*, replace: bool = False) -> list[Trigger]:
    """The three price breakouts, each AND `guard_metric > 0.2`. Composites, not labelers."""
    g = f"AND guard_metric > {MOMENTUM_GUARD_MIN} (breakout_gbb's guard direction)"
    specs = [
        ("breakout_gm", _breakout_gm, ("high", "close", "guard_metric"), f"breakout {g}"),
        ("breakout_vol_gm", _breakout_vol_gm, ("high", "close", "rvol", "guard_metric"),
         f"breakout_vol {g}"),
        ("breakout_squeeze_gm", _breakout_squeeze_gm,
         ("high", "close", "bb_width", "guard_metric"), f"breakout_squeeze {g}"),
    ]
    return [
        register(Trigger(name=n, evaluate=f, required_columns=c,
                         source="composite (023 Addendum 4)", description=d, **_fam(n)),
                 replace=replace)
        for n, f, c, d in specs
    ]


# -------------------------------------------------------------------------------------------------
# Breakout recency (feature 023, Addendum 7, 2026-10-11) -- from the Addendum 6 timing screen
# -------------------------------------------------------------------------------------------------

#: How many bars after a break still count as "recent". Matches the screen's k <= 8 window.
RECENT_BREAK_BARS = 8


def _break_recency(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(k, held) per bar: bars since the most recent 20-bar breakout (large if none yet), and
    whether every close since that break stayed above the broken level. Causal: only bars up to
    and including t are read."""
    close = _col(df, "close")
    prior_high = pd.Series(_col(df, "high")).rolling(BREAKOUT_LOOKBACK).max().shift(1).to_numpy()
    brk = _prior_high_breakout(df, BREAKOUT_LOOKBACK)
    n = len(close)
    k = np.full(n, 10**6, dtype=np.int64)
    held = np.zeros(n, dtype=bool)
    last, level, min_close = -1, np.nan, np.inf
    for t in range(n):
        if brk[t]:
            last, level, min_close = t, prior_high[t], close[t]
        if last >= 0:
            min_close = min(min_close, close[t])
            k[t] = t - last
            held[t] = min_close > level
    return k, held


def _breakout_gbb_nofresh(df: pd.DataFrame) -> pd.Series:
    from research.outcomes.builtin_triggers import _breakout_gbb
    k, held = _break_recency(df)
    fresh = (k == 0) | ((k <= RECENT_BREAK_BARS) & held)
    return pd.Series(_breakout_gbb(df).to_numpy() & ~fresh, index=df.index)


def _failed_breakout(df: pd.DataFrame) -> pd.Series:
    k, held = _break_recency(df)
    return pd.Series((k >= 1) & (k <= RECENT_BREAK_BARS) & ~held, index=df.index)


def register_break_recency_triggers(*, replace: bool = False) -> list[Trigger]:
    """Addendum 6's screen as triggers: drop fresh/held breaks from breakout_gbb; fade failures."""
    specs = [
        ("breakout_gbb_nofresh", _breakout_gbb_nofresh,
         ("high", "close", "guard_metric", "bb_width"),
         f"breakout_gbb AND NOT (break bar, or a break <= {RECENT_BREAK_BARS} bars ago that "
         f"has held above its level)"),
        ("failed_breakout", _failed_breakout, ("high", "close"),
         f"a 20-bar breakout 1..{RECENT_BREAK_BARS} bars ago with a close since back at or "
         f"below the broken level -- a REVERSION entry"),
    ]
    return [
        register(Trigger(name=n, evaluate=f, required_columns=c,
                         source="composite (023 Addendum 7)", description=d, **_fam(n)),
                 replace=replace)
        for n, f, c, d in specs
    ]
