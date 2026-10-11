"""The three required triggers (T023, FR-013, FR-014a).

`empty`, `gbb` and `geometry`. Kept out of `triggers.py` so the registry MECHANISM and the
registry CONTENT stay separable: a change to what `gbb` means must not touch the module
that enforces the no-outcome and no-pair constraints.

Each condition below was read off the LIVE `Framework/TrainingSignals.py`, not transcribed
from the spec. That mattered twice: `gbb` carries an rvol clause and a NaN convention the
spec's one-line summary omits, and `geometry` turned out not to be what the spec describes
at all (see `GEOMETRY_FINDINGS`).

THE TRIGGER/OUTCOME SPLIT, which is the whole point. A `LabelMethod` mixes a
backward-looking condition with its own forward-looking outcome test. Only the first half
is a trigger; the second half is what feature 021 replaces with the shared
`__outcome_positive`. For `labels_gbb` the split is clean and mechanical:

    buy_mask = (guard_metric < 0.3) & (bb_width > 0.035) & (rvol > -1.0)   <- TRIGGER
               & (future_gain >= 0.01)                                      <- OUTCOME

so porting it is a matter of dropping the last clause. Not every member splits that
cleanly, which is T025's audit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.outcomes.triggers import SOURCE_NATIVE, Trigger, register

# =================================================================================================
# empty — the verdict null (FR-014a)
# =================================================================================================

#: `labels_forward_return_mae_cap` (forward_mae) applies no entry condition at all: every
#: bar is a candidate and the label is decided purely by the forward outcome. Under this
#: design that IS a trigger -- the one that fires everywhere -- which is why the arm that
#: lost to cash and the verdict rule's null are the same object.
EMPTY_DESCRIPTION = (
    "every bar; the forward_mae case, and the null every other trigger is scored against"
)


def _empty(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


# =================================================================================================
# gbb — a faithful port of labels_gbb's backward-looking half
# =================================================================================================

#: Verbatim from `labels_gbb`'s signature. 0.3 matches the live entry gate
#: (`BaseStrategy.populate_entry_trend`: `guard_metric < entry_guard_threshold`); 0.035 is
#: DELIBERATELY not raised to the gate's 0.06, because matching it halves the label
#: (129,196 -> 65,107 bars) while admitting the identical 1,053 gate-passing bars.
GBB_GUARD_THRESHOLD = 0.3
GBB_BB_WIDTH_THRESHOLD = 0.035

#: `labels_gbb`'s default, and a NO-OP: rvol is non-negative and NaN becomes 0.0, so
#: `rvol > -1.0` admits every bar. Carried anyway rather than dropped, because dropping it
#: would silently change what this trigger means the moment anyone sets
#: `LABEL_RVOL_THRESHOLD` -- the clause is inert, not absent.
GBB_RVOL_THRESHOLD = -1.0


def _gbb(df: pd.DataFrame) -> pd.Series:
    """`labels_gbb`'s mask MINUS its `future_gain >= min_gain` clause.

    The NaN convention is `labels_gbb`'s, reproduced exactly rather than tidied, and it is
    ASYMMETRIC in a way that matters:

        guard_metric: NaN -> 0.0, and 0.0 < 0.3 is TRUE  -> a NaN guard_metric FIRES
        bb_width:     NaN -> 0.0, and 0.0 > 0.035 is FALSE -> a NaN bb_width does NOT

    So the two conditions disagree about what missing data means, and the combined mask
    does not fire on a bar missing `bb_width` but does fire on one missing only
    `guard_metric`. Writing the obvious `df["guard_metric"] < 0.3` instead would make a
    NaN compare False and silently diverge from the live label on exactly those bars.
    """
    n = len(df)
    guard = np.nan_to_num(np.asarray(df["guard_metric"], dtype=float), nan=0.0)
    bb_width = np.nan_to_num(np.asarray(df["bb_width"], dtype=float), nan=0.0)
    rvol = np.nan_to_num(
        np.asarray(df.get("rvol", pd.Series(np.zeros(n), index=df.index)), dtype=float),
        nan=0.0,
    )
    mask = (
        (guard < GBB_GUARD_THRESHOLD)
        & (bb_width > GBB_BB_WIDTH_THRESHOLD)
        & (rvol > GBB_RVOL_THRESHOLD)
    )
    return pd.Series(mask, index=df.index)


# =================================================================================================
# geometry — NOT a port, and the reason is a finding
# =================================================================================================

#: Read `labels_geometry` (TrainingSignals.py:2655) before trusting the spec's description
#: of it. FR-013 calls it "the DWT-smoothed zero-lag geometric valley condition already
#: implemented in `labels_geometry` but never tested". Four things are wrong with that, all
#: verified by running the function:
#:
#:   1. IT CANNOT BE CALLED WITH ITS OWN DEFAULTS. `prominence` defaults to None, which
#:      sends it through `min(price_std, price_mean * min_gain) * 0.5` with `min_gain` also
#:      defaulting to None -- `float * None` raises TypeError. "Never tested" is literal:
#:      the function has never run without a caller supplying `min_gain`.
#:   2. THE DWT SMOOTHING IS DISABLED. `smoothed = _dwt_smooth(close)` is commented out and
#:      replaced by `smoothed = close`. `_dwt_smooth` is defined and importable
#:      (TrainingSignals.py:2619); it is simply not called. There is no "DWT-smoothed
#:      zero-lag approximation" in the live code.
#:   3. IT LOOKS FORWARD, so it is not a trigger. `find_peaks` runs over the WHOLE close
#:      array, and a local minimum is only identifiable once later bars confirm it; the
#:      loop then also labels the THREE BARS PRECEDING each valley. A bar is therefore
#:      labelled on the strength of what happens up to three bars after it.
#:   4. ITS THRESHOLD IS PAIR-DEPENDENT AND WHOLE-SERIES. `prominence` is derived from
#:      `np.mean` and `np.std` over the entire series, so it differs per pair and uses the
#:      future -- which is both lookahead and, in effect, a per-pair override
#:      (Constitution X).
#:
#: Measured for scale: with `min_gain=0.01` on 500 synthetic bars it fires on 272 of them
#: (54%), because it marks four bars per valley at `distance=4`. That is not a selective
#: valley condition.
#:
#: CONSEQUENCE. Registering it faithfully would put a lookahead-bearing trigger on the
#: leaderboard, where it would collect credit for information unavailable at entry and
#: show inflated economic value -- exactly the failure Constitution V and FR-009 exist to
#: prevent, and it would discredit the one artefact this feature is built to produce. So
#: `geometry` below is a CAUSAL valley condition in the same spirit, not a port, and this
#: divergence is a reporting obligation for T025 and the leaderboard.
GEOMETRY_FINDINGS = (
    "labels_geometry is not portable as a trigger: it raises with its own defaults, its "
    "DWT smoothing is commented out, find_peaks plus a 3-bar backfill makes it "
    "forward-looking, and its prominence is a whole-series per-pair statistic. The "
    "registered `geometry` trigger is a causal substitute, not a port."
)

#: Bars after the low at which the turn is considered confirmed. Mapped from
#: `labels_geometry`'s own `distance=horizon // 2` with its `horizon = 9`, i.e. 4.
GEOMETRY_CONFIRM_BARS = 4

#: Minimum fall into the low, as a fraction, standing in for `find_peaks`' `prominence`.
#: A RELATIVE depth deliberately, so it means the same thing on a $0.50 pair and a $60,000
#: one -- unlike the live absolute `min(std, mean*min_gain)*0.5`, which is a per-pair
#: quantity. 1% matches the `min_gain=0.01` every caller of labels_geometry has had to pass.
GEOMETRY_MIN_DEPTH = 0.01


def _geometry(df: pd.DataFrame) -> pd.Series:
    """A CAUSAL confirmed-valley condition. Fires `GEOMETRY_CONFIRM_BARS` after the low.

    At bar `i`, with `k = GEOMETRY_CONFIRM_BARS`, fires when all three hold using only
    bars at or before `i`:

      * the lowest close in the trailing `2k + 1` bars sits at `i - k` -- a local minimum
        with `k` bars either side of it, every one of them already observed;
      * the fall into that low, measured from the highest close in the window before it,
        is at least `GEOMETRY_MIN_DEPTH` (the prominence analogue);
      * the close has since risen above the low, so the turn is confirmed rather than
        assumed.

    It therefore fires `k` bars LATE relative to `labels_geometry`, which marks the valley
    itself and the three bars before it. That lateness is the honest part: a valley is not
    knowable at its own bar, and an entry taken there is an entry no strategy could have
    taken. Whether a trigger that fires after the turn still selects collectable moments
    is precisely the question the leaderboard exists to answer.
    """
    k = GEOMETRY_CONFIRM_BARS
    w = 2 * k + 1
    close = pd.to_numeric(df["close"], errors="coerce")

    # argmin of the trailing window, as an offset from the window's start. The window ends
    # at the current bar, so every value in it is already observed.
    pos_of_min = close.rolling(w).apply(np.argmin, raw=True)
    is_valley_at_k = pos_of_min == k

    low = close.shift(k)
    # Highest close in the part of the window BEFORE the low, i.e. the fall into it.
    prior_high = close.shift(k + 1).rolling(k).max()
    deep_enough = (prior_high - low) / low >= GEOMETRY_MIN_DEPTH
    turned = close > low

    mask = is_valley_at_k & deep_enough & turned
    # The warm-up (first 2k+1 bars) is genuinely unevaluable; `fillna(False)` is explicit
    # here rather than left to pandas, because `triggers.evaluate` rejects a mask that
    # still carries missingness and the author must choose, not the framework.
    return mask.fillna(False).astype(bool)



# =================================================================================================
# Trigger families (feature 024 FR-001) -- what each condition TESTS, not what it is called
# =================================================================================================

_FAMILY = {
    "empty": (
        "unassigned",
        "fires on every bar: no direction (the null every trigger is scored against)",
    ),
    "gbb": ("reversion", "guard_metric < 0.3 with a wide band: buys LOW band position (weakness)"),
    "geometry": ("reversion", "a confirmed valley after a >=1% local low: buys a dip"),
    "adx": (
        "unassigned",
        "adx_scaled > -0.2 passes ~62% of bars: a trend-strength filter with no direction",
    ),
    "breakout": ("momentum", "close above the prior 20-bar high: buys a new high (strength)"),
    "breakout_gbb": (
        "momentum",
        "guard_metric > 0.2 with a wide band: buys HIGH band position (strength)",
    ),
    "gbb_tight": ("reversion", "gbb's weakness condition with tightened thresholds"),
    "random_matched": ("unassigned", "pseudo-random control: carries no market information"),
    "gbb_stale": (
        "reversion",
        "gbb lagged 1500 bars: the same weakness condition, information stale (control)",
    ),
}


def _fam(name: str) -> dict:
    family, reason = _FAMILY[name]
    return {"family": family, "family_reason": reason}

# =================================================================================================
# Registration
# =================================================================================================

def register_builtins(*, replace: bool = False) -> list[Trigger]:
    """Register the three FR-013 members. Idempotent via `replace`."""
    return [
        register(Trigger(
            name="empty",
            **_fam("empty"),
            evaluate=_empty,
            required_columns=(),
            source="labels_forward_return_mae_cap",
            description=EMPTY_DESCRIPTION,
        ), replace=replace),
        register(Trigger(
            name="gbb",
            **_fam("gbb"),
            evaluate=_gbb,
            required_columns=("guard_metric", "bb_width"),
            source="labels_gbb",
            description=(
                f"guard_metric < {GBB_GUARD_THRESHOLD} AND bb_width > "
                f"{GBB_BB_WIDTH_THRESHOLD} (rvol clause inert at its default)"
            ),
        ), replace=replace),
        register(Trigger(
            name="geometry",
            **_fam("geometry"),
            evaluate=_geometry,
            required_columns=("close",),
            source=SOURCE_NATIVE,
            description=(
                f"causal confirmed valley, {GEOMETRY_CONFIRM_BARS} bars after a local low "
                f"at least {GEOMETRY_MIN_DEPTH:.0%} deep — NOT a port of labels_geometry; "
                f"see GEOMETRY_FINDINGS"
            ),
        ), replace=replace),
    ]


# =================================================================================================
# Derived from T025's audit. Each is the separable BACKWARD factor of one or more LabelMethods,
# verified by containment on real data: every label those methods produce falls inside its
# trigger (SOL/USDT, 164,918 bars, 11 labeler/trigger pairs, zero violations).
# =================================================================================================

#: `labels_future_sharpe` / `_sortino` / `_expectancy` all end with
#:     labels = np.where(adx_scaled > -0.2, labels, 0)
#: described in the live code as filtering out sideways markets. Research R7 predicted these
#: three were purely forward-path and so NOT reducible; they are, and the clause is BINDING,
#: not decorative -- it blocks 38.35% of bars pooled over five pairs (28.5% on AAVE to 48.8%
#: on XRP), so it cannot be dismissed as inert the way `gbb`'s rvol clause can.
#:
#: Why it is a trigger and not a labeler detail: it is a pure function of the current bar's
#: `adx_scaled`. Under this design that makes it exactly the kind of condition the leaderboard
#: should PRICE rather than leave buried in three label functions -- its economic value and
#: learnability are measurable, so whether it earns its keep stops being a judgement call.
ADX_THRESHOLD = -0.2


def _adx(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.nan_to_num(np.asarray(df["adx_scaled"], dtype=float), nan=0.0) > ADX_THRESHOLD,
        index=df.index,
    )


#: `labels_breakout`'s entry condition, shared verbatim by `_tb`, `_vol` and `_squeeze`:
#:     prior_high = high.rolling(lookback).max().shift(1);  close > prior_high
#: Cleanly causal -- `rolling` plus `shift(1)` reads only closed bars -- so unlike `geometry`
#: this one ports exactly. `lookback` is `labels_breakout`'s own default.
#:
#: Carries this repository's loudest standing warning: the breakout label is LEARNABLE but
#: NOT TRADEABLE (MCC 0.77 while losing money on every timeframe and exit tested), and a
#: re-test after the exit defect was fixed still returned -17.55%. That makes it the single
#: most valuable row on the leaderboard: a trigger already known to lose money is the
#: sharpest available test of whether the economic-value column can tell the difference.
BREAKOUT_LOOKBACK = 20


def _breakout(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    prior_high = high.rolling(BREAKOUT_LOOKBACK).max().shift(1)
    out = pd.to_numeric(df["close"], errors="coerce") > prior_high
    # Warm-up (first `lookback` bars) has no prior high; filled explicitly by this trigger,
    # because `triggers.evaluate` refuses a mask that still carries missingness.
    return out.fillna(False).astype(bool)


#: `labels_breakout_gbb`'s factor. Note the DIRECTION: `guard_metric > 0.2`, the OPPOSITE of
#: `gbb`'s `guard_metric < 0.3`. The two families named "gbb" therefore select opposite
#: guard_metric regimes while sharing the `bb_width` floor, which is worth knowing before
#: reading any leaderboard row that mentions gbb -- they are not variants of one condition.
BREAKOUT_GBB_GUARD_THRESHOLD = 0.2
BREAKOUT_GBB_BB_WIDTH_THRESHOLD = 0.035


def _breakout_gbb(df: pd.DataFrame) -> pd.Series:
    guard = np.nan_to_num(np.asarray(df["guard_metric"], dtype=float), nan=0.0)
    bb_width = np.nan_to_num(np.asarray(df["bb_width"], dtype=float), nan=0.0)
    mask = (guard > BREAKOUT_GBB_GUARD_THRESHOLD) & (
        bb_width > BREAKOUT_GBB_BB_WIDTH_THRESHOLD
    )
    return pd.Series(mask, index=df.index)


def register_audit_triggers(*, replace: bool = False) -> list[Trigger]:
    """The distinct backward factors T025's audit found (FR-012).

    Registered as DISTINCT CONDITIONS, not one per labeler: `gbb` already covers
    `labels_gbb`/`_path`/`_fwdmae`, and `breakout` covers four breakout labelers, so one
    trigger per labeler would pad the leaderboard with duplicate rows and make the verdict
    column harder to read, not easier.
    """
    return [
        register(Trigger(
            name="adx",
            **_fam("adx"),
            evaluate=_adx,
            required_columns=("adx_scaled",),
            source="labels_future_sharpe/_sortino/_expectancy",
            description=(
                f"adx_scaled > {ADX_THRESHOLD} — binding, blocks 38.35% of bars; "
                f"refutes research R7's 'not reducible' prediction for these three"
            ),
        ), replace=replace),
        register(Trigger(
            name="breakout",
            **_fam("breakout"),
            evaluate=_breakout,
            required_columns=("high", "close"),
            source="labels_breakout/_tb/_vol/_squeeze",
            description=(
                f"close > prior {BREAKOUT_LOOKBACK}-bar high (causal); known LEARNABLE but "
                f"NOT TRADEABLE, so the sharpest test of the economic-value column"
            ),
        ), replace=replace),
        register(Trigger(
            name="breakout_gbb",
            **_fam("breakout_gbb"),
            evaluate=_breakout_gbb,
            required_columns=("guard_metric", "bb_width"),
            source="labels_breakout_gbb",
            description=(
                f"guard_metric > {BREAKOUT_GBB_GUARD_THRESHOLD} AND bb_width > "
                f"{BREAKOUT_GBB_BB_WIDTH_THRESHOLD} — guard direction is OPPOSITE to gbb's"
            ),
        ), replace=replace),
    ]


# =================================================================================================
# gbb_tight — the margin finding as a trigger (2026-10-09)
# =================================================================================================

#: `gbb`'s own condition with both thresholds tightened. Registered because the 2026-10-09
#: diagnostic found the economic value sits in the trigger's MARGIN rather than its binary
#: firing: EV over gbb bars rises monotonically with threshold depth, from +0.000822 at
#: 100% of gbb bars to +0.050938 at 0.068%, and at matched selectivity a bare threshold beat
#: the production model's own selection 5.6x (+0.029014 against +0.005204).
#:
#: HOW THESE NUMBERS WERE CHOSEN, and why not the best ones. The rung ladder was explored BY
#: EV, so picking the rung with the highest EV would make this row's own leaderboard entry
#: self-fulfilling -- selection on the outcome being measured. The thresholds below are
#: instead chosen on a criterion INDEPENDENT of EV: they are the rung whose selectivity most
#: nearly MATCHES THE ARM'S (0.389% of gbb bars against the arm's 0.470%), because the
#: question this trigger exists to ask is "can a plain threshold, at the model's own
#: selectivity, do what the model does".
#:
#: READ ITS EV WITH THAT IN MIND ANYWAY. Selectivity-matching removes the worst of the
#: selection effect, not all of it: the ladder's shape was known before these values were
#: picked. The row is honest as a comparison against the ARM at equal selectivity; it is not
#: an unbiased estimate of what an independently-chosen tight threshold would score.
#:
#: AND IT IS NOT A PROMOTION CANDIDATE. Table EV is a lookahead bound, not a return
#: (FR-017), and this repository has twice measured "ceiling rises monotonically, return
#: doesn't" -- once for trigger relaxation, once for breakout tail labelling. At 0.389% of
#: gbb bars the trade count also falls below what the pooled harness resolves at ~3-4pp/yr.
#: Only `evaluation/` can say whether this trades.
GBB_TIGHT_GUARD_THRESHOLD = -0.8
GBB_TIGHT_BB_WIDTH_THRESHOLD = 0.18


def _gbb_tight(df: pd.DataFrame) -> pd.Series:
    """`_gbb`'s arithmetic with the tightened thresholds, NaN convention included.

    Deliberately not implemented as `_gbb(df) & extra_condition`: the thresholds are what
    differ, so expressing it as a conjunction with gbb's own would apply gbb's looser
    thresholds first and then a second pair, which is the same mask by luck rather than by
    construction. Sharing the NaN handling matters more -- a NaN `guard_metric` becomes 0.0
    and so FAILS this trigger's `< -0.8` while it PASSES gbb's `< 0.3`, and that asymmetry
    should come from one place.
    """
    n = len(df)
    guard = np.nan_to_num(np.asarray(df["guard_metric"], dtype=float), nan=0.0)
    bb_width = np.nan_to_num(np.asarray(df["bb_width"], dtype=float), nan=0.0)
    rvol = np.nan_to_num(
        np.asarray(df.get("rvol", pd.Series(np.zeros(n), index=df.index)), dtype=float),
        nan=0.0,
    )
    mask = (
        (guard < GBB_TIGHT_GUARD_THRESHOLD)
        & (bb_width > GBB_TIGHT_BB_WIDTH_THRESHOLD)
        & (rvol > GBB_RVOL_THRESHOLD)
    )
    return pd.Series(mask, index=df.index)


def register_margin_triggers(*, replace: bool = False) -> list[Trigger]:
    return [
        register(Trigger(
            name="gbb_tight",
            **_fam("gbb_tight"),
            evaluate=_gbb_tight,
            required_columns=("guard_metric", "bb_width"),
            source="labels_gbb (thresholds tightened)",
            description=(
                f"guard_metric < {GBB_TIGHT_GUARD_THRESHOLD} AND bb_width > "
                f"{GBB_TIGHT_BB_WIDTH_THRESHOLD} — selectivity matched to the arm's 0.470% "
                f"of gbb bars, NOT chosen for best EV; not a promotion candidate"
            ),
        ), replace=replace),
    ]


# =================================================================================================
# CONTROLS (2026-10-09, after T035 found the empty-vs-itself control vacuous)
# =================================================================================================

#: `empty` vs ITSELF was the board's only control and it cannot fail. Identical masks give
#: identical per-block sums, so the paired bootstrap returns [0, 0] by construction — it
#: proves the subtraction works, not that the reference is wired to the right trigger or that
#: the pipeline has no bias. A control that cannot fail is worthless, which is the same
#: standard this feature applied to its duplication guard (T005 plants a duplicate to prove
#: the detector fires).
#:
#: `random_matched` is a real control: a pseudo-random mask at a base rate matched to `gbb`,
#: carrying NO information about the market at all. It MUST come out indeterminate, and its
#: economic value MUST sit at the `empty` null within the interval. If it does not, the
#: pipeline is manufacturing an effect from the act of selecting a subset — exactly the kind
#: of defect the signal-bar leak was, and exactly what no self-comparison could have caught.
RANDOM_CONTROL_BASE_RATE = 0.17871        # gbb's pooled base rate
RANDOM_CONTROL_SEED = 20261009


def _random_matched(df: pd.DataFrame) -> pd.Series:
    """A market-independent mask at `gbb`'s base rate.

    Seeded from the BAR COUNT only, never from prices, so it is deterministic and
    reproducible per pair while carrying no market information. Deliberately not seeded from
    the date index either: a date-derived mask could correlate with time-of-day effects,
    which would make the control itself informative and defeat its purpose.
    """
    rng = np.random.default_rng(RANDOM_CONTROL_SEED + len(df))
    return pd.Series(rng.random(len(df)) < RANDOM_CONTROL_BASE_RATE, index=df.index)


#: A trigger whose signal is shifted far beyond any hold, so whatever it knew about the
#: market is stale by the time it is paired with an outcome. `max_hold_days` is 10 days =
#: 960 bars at 15m, so 1,500 bars is comfortably past it. Should behave like
#: `random_matched`: present at gbb's base rate, carrying nothing usable.
STALE_CONTROL_LAG_BARS = 1500


def _gbb_stale(df: pd.DataFrame) -> pd.Series:
    """`gbb`, lagged 1,500 bars. A second, independent control.

    `random_matched` tests "does selecting a subset at all create an effect"; this tests
    "does selecting a subset WITH GBB'S OWN CLUSTERING STRUCTURE create one". The two fail
    differently: a bias from the sampling machinery shows in the first, a bias from
    autocorrelated clustering shows in the second. The block bootstrap's whole purpose is to
    price the second, so it needs a case where the right answer is known to be zero.
    """
    fired = _gbb(df).to_numpy()
    out = np.zeros_like(fired)
    if len(fired) > STALE_CONTROL_LAG_BARS:
        out[STALE_CONTROL_LAG_BARS:] = fired[:-STALE_CONTROL_LAG_BARS]
    return pd.Series(out, index=df.index)


def register_controls(*, replace: bool = False) -> list[Trigger]:
    """Two controls whose correct verdict is known in advance to be `indeterminate`."""
    return [
        register(Trigger(
            name="random_matched", evaluate=_random_matched, required_columns=(),
            **_fam("random_matched"),
            source="CONTROL",
            description=(
                f"pseudo-random at gbb's {RANDOM_CONTROL_BASE_RATE:.3%} base rate; carries "
                f"no market information. MUST be indeterminate and sit at the null."
            ),
        ), replace=replace),
        register(Trigger(
            name="gbb_stale", evaluate=_gbb_stale, required_columns=("guard_metric", "bb_width"),
            **_fam("gbb_stale"),
            source="CONTROL",
            description=(
                f"gbb lagged {STALE_CONTROL_LAG_BARS} bars (> max_hold's 960), so its "
                f"information is stale but its clustering is intact. MUST be indeterminate."
            ),
        ), replace=replace),
    ]
