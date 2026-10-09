"""Economic value from the table alone (T027, FR-014, FR-009b).

The tests that matter here are the ones about CLUSTERING. The all-bars mean is the quantity
FR-014 literally specifies ("the economic value of trading every triggered bar"), and it is
not comparable across triggers: a trigger whose bars arrive in clusters re-samples the same
episode many times, so its mean is weighted by how often it repeats itself rather than by
how good its opportunities are. Measured on real data, correcting for that FLIPS `gbb` from
positive to negative. So the non-overlapping figure is tested as carefully as the headline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "utils"), str(ROOT / "NNNC")):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.outcomes import builtin_triggers as B  # noqa: E402
from research.outcomes import leaderboard as LB  # noqa: E402
from research.outcomes import triggers as T  # noqa: E402
from research.outcomes.leakage import (  # noqa: E402
    OUTCOME_COLUMN_CLOSE_REASON,
    OUTCOME_COLUMN_FILLABLE,
    OUTCOME_COLUMN_GAPFILL_SHARE,
    OUTCOME_COLUMN_HOLD_BARS,
    OUTCOME_COLUMN_POSITIVE,
    OUTCOME_COLUMN_PROFIT,
)


@pytest.fixture(autouse=True)
def _registry():
    T.clear_registry()
    B.register_builtins()
    yield
    T.clear_registry()


def _frames(profit, hold, *, reason=None, fillable=None, n=None):
    """Builds n+1 bars with a PREPENDED signal-only bar.

    Since 2026-10-09 a trigger on bar i is paired with outcome row i+1 (`labels.entry_mask`):
    a trigger reads its bar's close, so the earliest it can transact is the next bar's open.
    So bar 0 is never an entry. Prepending one bar keeps the SCORED population exactly the
    `profit`/`hold` arrays each test specifies, instead of every expectation needing a
    manual -1. The prepended row's own outcome is unreachable by construction.
    """
    n = n or len(profit)
    profit = [0.0] + list(profit)
    hold = [0] + list(hold)
    if reason is not None:
        reason = ["stop"] + list(reason)
    if fillable is not None:
        fillable = [True] + list(fillable)
    n = n + 1
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    feat = pd.DataFrame({"date": dates, "close": np.arange(n, dtype=float) + 100.0})
    out = pd.DataFrame({
        "date": dates,
        OUTCOME_COLUMN_PROFIT: np.asarray(profit, dtype=float),
        OUTCOME_COLUMN_HOLD_BARS: np.asarray(hold, dtype=float),
        OUTCOME_COLUMN_CLOSE_REASON: reason if reason is not None else ["stop"] * n,
        OUTCOME_COLUMN_FILLABLE: (
            np.asarray(fillable, dtype=bool) if fillable is not None
            else np.ones(n, dtype=bool)
        ),
        OUTCOME_COLUMN_GAPFILL_SHARE: np.zeros(n),
        OUTCOME_COLUMN_POSITIVE: np.asarray(profit, dtype=float) > 0.0,
    })
    return feat, out


# =============================================================================================
# The headline statistic
# =============================================================================================

def test_economic_value_is_the_mean_realised_outcome_over_triggered_bars():
    feat, out = _frames([0.01, -0.02, 0.03, 0.04], [0, 0, 0, 0])
    s = LB.economic_value(feat, out, "empty")
    assert s.mean == pytest.approx(np.mean([0.01, -0.02, 0.03, 0.04]))
    assert s.n_scored == 4
    assert s.base_rate == 1.0
    assert s.win_rate == pytest.approx(0.75)


def test_no_model_is_involved_only_the_trigger_and_the_table():
    """The column's defining property: it must not consult `__outcome_positive`, a
    threshold, or anything trained. Every triggered bar is traded, winners and losers."""
    feat, out = _frames([-0.05] * 10, [0] * 10)
    s = LB.economic_value(feat, out, "empty")
    assert s.mean == pytest.approx(-0.05), "losers must be included, not filtered out"
    assert s.win_rate == 0.0


def test_end_of_data_rows_are_excluded_rather_than_counted_as_zero():
    """`simulate_forward` returns 0.0 for a window that ran off the series end. That is a
    FABRICATED zero, not a realised outcome, and averaging it in pulls the mean toward
    zero by its share."""
    feat, out = _frames([0.10, 0.0, 0.10], [0, 0, 0],
                        reason=["stop", "end_of_data", "stop"])
    s = LB.economic_value(feat, out, "empty")
    assert s.n_scored == 2
    assert s.n_excluded_end_of_data == 1
    assert s.mean == pytest.approx(0.10), "the fabricated zero must not dilute the mean"


def test_distribution_is_reported_not_just_the_mean():
    """FR-014 requires the distribution because the mean cannot distinguish 'many small
    wins and a few catastrophes' from 'a steady small edge' -- and those call for opposite
    decisions. Measured on the real table: mean -0.001427 against median +0.005595."""
    feat, out = _frames([0.01] * 95 + [-1.0] * 5, [0] * 100)
    s = LB.economic_value(feat, out, "empty")
    assert s.mean < 0.0 < s.median, "the fixture must reproduce the real asymmetry"
    assert set(s.quantiles) == set(LB.QUANTILES)
    assert s.quantiles[0.05] < s.quantiles[0.95]


# =============================================================================================
# Clustering — the correction that changes the ranking
# =============================================================================================

def test_non_overlapping_selection_skips_bars_inside_an_open_trade():
    """A bar at index i with hold h occupies [i, i+h]. The greedy scan takes a trade then
    skips to i+h+1, so five consecutive bars holding 2 bars each yield two trades."""
    feat, out = _frames([0.01] * 5, [2] * 5)
    s = LB.economic_value(feat, out, "empty")
    assert s.n_scored == 5
    assert s.n_non_overlapping == 2          # bars 0 and 3
    assert s.cluster_ratio == pytest.approx(2 / 5)


def test_clustering_can_FLIP_the_sign_of_the_economic_value():
    """THE reason the non-overlapping figure exists, and it is not hypothetical: on real
    data this correction moves `gbb` from +0.000822 to -0.001262, from second place to
    last, below the null.

    Here a single bad opportunity is sampled once while a good one is re-sampled four
    times, so the all-bars mean is positive and the comparable mean is negative."""
    profit = [-0.10] + [0.04, 0.04, 0.04, 0.04]
    hold = [0, 3, 3, 3, 3]          # the four winners all describe ONE episode
    feat, out = _frames(profit, hold)
    s = LB.economic_value(feat, out, "empty")
    assert s.mean > 0.0, "all-bars mean is inflated by the repeated episode"
    assert s.mean_non_overlapping < 0.0, "one opportunity each: the edge disappears"


def test_greedy_selection_is_earliest_first_not_best_first():
    """Greedy EARLIEST is deliberate. Choosing the best non-overlapping subset would be a
    lookahead selection of its own and would flatter every trigger; the point is to count
    distinct OPPORTUNITIES, not to construct a favourable schedule."""
    feat, out = _frames([0.01, 0.50], [1, 0])
    s = LB.economic_value(feat, out, "empty")
    assert s.n_non_overlapping == 1
    assert s.mean_non_overlapping == pytest.approx(0.01), "took bar 0, not the better bar 1"


def test_zero_hold_bars_do_not_suppress_the_following_bar():
    feat, out = _frames([0.01] * 4, [0] * 4)
    assert LB.economic_value(feat, out, "empty").n_non_overlapping == 4


# =============================================================================================
# Fillable-only (FR-009b)
# =============================================================================================

def test_fillable_only_restricts_the_population():
    """Not a rounding adjustment: only 28.17% of real bars are fillable. Omitting it would
    let a trigger earn its value on bars no one could have traded -- the failure that
    inflated an out-of-sample result here until the volume filter was turned on."""
    feat, out = _frames([0.10, -0.10, 0.10, -0.10], [0] * 4,
                        fillable=[True, False, True, False])
    s = LB.economic_value(feat, out, "empty", fillable_only=True)
    assert s.n_scored == 2
    assert s.mean == pytest.approx(0.10)
    assert LB.economic_value(feat, out, "empty").mean == pytest.approx(0.0)


def test_base_rate_is_over_all_bars_even_when_fillable_filtered():
    """The base rate describes the TRIGGER, not the fillable subset; conflating them would
    make a trigger look rarer than it is on an illiquid pair."""
    feat, out = _frames([0.1] * 10, [0] * 10, fillable=[True] * 2 + [False] * 8)
    assert LB.economic_value(feat, out, "empty", fillable_only=True).base_rate == 1.0


def test_fillable_only_without_the_column_is_a_clear_error():
    feat, out = _frames([0.1], [0])
    with pytest.raises(KeyError, match="fillable"):
        LB.economic_value(feat, out.drop(columns=[OUTCOME_COLUMN_FILLABLE]), "empty",
                          fillable_only=True)


# =============================================================================================
# Pooling
# =============================================================================================

def test_pooling_is_on_the_bar_not_an_average_of_per_pair_means():
    """A per-pair average weights a small pair equally with a large one, which would hide
    the concentration this repository's edge is already known to have."""
    a = _frames([0.10] * 90, [0] * 90)
    b = _frames([-0.10] * 10, [0] * 10)
    pooled = LB.pooled_economic_value([a, b], "empty")
    assert pooled.n_scored == 100
    assert pooled.mean == pytest.approx((90 * 0.10 + 10 * -0.10) / 100)
    naive = (LB.economic_value(*a, "empty").mean + LB.economic_value(*b, "empty").mean) / 2
    assert pooled.mean != pytest.approx(naive), "bar-pooling must differ from pair-averaging"


def test_the_non_overlapping_scan_does_not_run_ACROSS_pairs():
    """A long trade at the end of one pair must not suppress the first bar of the next.
    Pooling the concatenation instead of the per-pair selections would do exactly that."""
    a = _frames([0.01], [1000])          # one very long trade
    b = _frames([0.02], [0])
    pooled = LB.pooled_economic_value([a, b], "empty")
    assert pooled.n_non_overlapping == 2, "pair b's bar was suppressed by pair a's trade"


def test_pooling_requires_frames():
    with pytest.raises(ValueError, match="no frames"):
        LB.pooled_economic_value([], "empty")


# =============================================================================================
# Guards
# =============================================================================================

def test_misaligned_frames_are_refused():
    feat, out = _frames([0.1] * 5, [0] * 5)
    with pytest.raises(Exception):
        LB.economic_value(feat, out.iloc[:-1], "empty")


def test_a_missing_profit_column_is_a_clear_error():
    feat, out = _frames([0.1], [0])
    with pytest.raises(KeyError, match="build the table first"):
        LB.economic_value(feat, out.drop(columns=[OUTCOME_COLUMN_PROFIT]), "empty")


def test_a_trigger_that_fires_on_nothing_returns_empty_stats_not_a_crash():
    feat, out = _frames([0.1] * 5, [0] * 5)
    T.register(T.Trigger("never", lambda d: pd.Series(False, index=d.index), ()))
    s = LB.economic_value(feat, out, "never")
    assert s.is_empty and s.n_scored == 0 and np.isnan(s.mean)


def test_header_and_row_widths_line_up():
    feat, out = _frames([0.1] * 5, [0] * 5)
    s = LB.economic_value(feat, out, "empty")
    assert len(LB.format_row("empty", s, s)) == len(LB.header())


# =============================================================================================
# The verdict rule (T030, FR-014b as amended by FR-015b, SC-007)
# =============================================================================================

def _iv(point, lo, hi):
    from research.outcomes.bootstrap import Interval
    return Interval(point, lo, hi, 0.05, n_blocks=100, block_length=48, n_resamples=1000)


@pytest.mark.parametrize("ev_lo,learn_delta,learn_se,expected", [
    # economic value clears its bar -> beats_null_fillable, regardless of learnability
    (+0.001, +0.50, 0.01, LB.VERDICT_BEATS_NULL_FILLABLE),
    (+0.001, +0.00, 0.01, LB.VERDICT_BEATS_NULL_FILLABLE),
    (+0.001, -0.50, 0.01, LB.VERDICT_BEATS_NULL_FILLABLE),
    # it does not, but learnability clears 1.96 SE -> decoration
    (-0.001, +0.50, 0.01, LB.VERDICT_DECORATION),
    (-0.001, +0.0197, 0.01, LB.VERDICT_DECORATION),
    # learnability falls just SHORT of the 1.96 bar it used to clear at 1.0 -- this is the
    # amended bar, and it is the case that moved votes_consensus off `decoration`
    (-0.001, +0.011, 0.01, LB.VERDICT_INDETERMINATE),
    # neither -> indeterminate
    (-0.001, +0.009, 0.01, LB.VERDICT_INDETERMINATE),
    (-0.001, -0.50, 0.01, LB.VERDICT_INDETERMINATE),
    (0.0, 0.0, 0.01, LB.VERDICT_INDETERMINATE),
])
def test_verdict_rule_branches(ev_lo, learn_delta, learn_se, expected):
    v = LB.assign_verdict("t", _iv(0.002, ev_lo, 0.01), learn_delta, learn_se)
    assert v.verdict == expected


def test_the_learnability_bar_uses_the_SE_OF_THE_DIFFERENCE():
    """FR-014b asks for "the SE on that difference". The old code used the trigger's own
    fold SE and never combined the reference's, understating the difference's SE by up to
    sqrt(2) and making `decoration` easier to earn than specified."""
    v = LB.assign_verdict("t", _iv(0.0, -0.001, 0.002), 0.0166, 0.0081, reference_se=0.0092)
    assert v.learn_se == pytest.approx((0.0081**2 + 0.0092**2) ** 0.5, rel=1e-9)
    # votes_consensus's real numbers: clears 1.0x its OWN SE, fails 1.96x the DIFFERENCE's
    assert 0.0166 > 0.0081
    assert v.verdict == LB.VERDICT_INDETERMINATE


def test_a_wholly_negative_interval_is_WORSE_THAN_NULL_not_indeterminate():
    """`indeterminate` was absorbing rows measured to be worse than the null, filing them
    with the controls. "Not shown to beat" and "shown to be worse" are different findings."""
    v = LB.assign_verdict("t", _iv(-0.0023, -0.0029, -0.0018), -0.001, 0.01, reference_se=0.01)
    assert v.verdict == LB.VERDICT_WORSE_THAN_NULL


def test_learnable_but_unprofitable_stays_DECORATION_even_when_negative():
    """Branch order matters: a trigger worse than the null on EV but genuinely learnable is
    still explanation (b), which is the more informative statement about it."""
    v = LB.assign_verdict("t", _iv(-0.0023, -0.0029, -0.0018), +0.50, 0.01, reference_se=0.01)
    assert v.verdict == LB.VERDICT_DECORATION


def test_the_rule_is_TOTAL_every_input_gets_exactly_one_verdict():
    """SC-007 demands totality. The third branch is unconditional, so this is structural --
    but it is asserted over a grid rather than argued, because "indeterminate MUST be
    reported as such rather than resolved by judgement" is the point of having it."""
    seen = set()
    for lo in (-0.01, -1e-12, 0.0, +1e-12, +0.01):
        for ld in (-1.0, -0.001, 0.0, 0.001, 1.0):
            for se in (1e-6, 0.01, 1.0):
                v = LB.assign_verdict("t", _iv(0.0, lo, 0.02), ld, se)
                assert v.verdict in {
                    LB.VERDICT_BEATS_NULL_FILLABLE, LB.VERDICT_DECORATION,
                    LB.VERDICT_INDETERMINATE, LB.VERDICT_WORSE_THAN_NULL,
                }
                seen.add(v.verdict)
    assert len(seen) == 3, f"grid never produced three verdicts: {seen}"


def test_an_interval_touching_zero_does_NOT_count_as_beating_the_null():
    """`lo > 0` strictly. An interval whose lower bound is exactly zero spans zero, and
    FR-014b requires exceeding the null by MORE than the uncertainty."""
    assert not LB.assign_verdict("t", _iv(0.001, 0.0, 0.01), 0.0, 1.0).ev_beats_null


def test_a_SIGNIFICANTLY_NEGATIVE_economic_value_never_beats_the_null():
    """gbb's real case on the fillable population: -0.004102 [-0.004793, -0.003402]. An
    interval wholly BELOW zero must never read as beating the null -- which a
    `not spans_zero` test would have got wrong.

    The VERDICT on this row changed with the 2026-10-09 amendment: at the old 1-SE bar its
    learnability (+0.0175 +/- 0.0109) cleared `decoration`; at 1.96 SE it does not
    (0.0175 < 0.0214), so it now files as `worse_than_null`. The invariant this test exists
    for -- never `beats_null_fillable` -- is unaffected either way."""
    v = LB.assign_verdict("gbb", _iv(-0.004102, -0.004793, -0.003402), +0.0175, 0.0109)
    assert not v.ev_beats_null
    assert v.verdict == LB.VERDICT_WORSE_THAN_NULL


def test_learnability_bar_is_1_96_SE_MATCHING_the_economic_value_column():
    """AMENDED 2026-10-09 (T035). Was one SE, as the contract words it; that made
    `decoration` a ~68% bar against the economic-value column's 95%, so it was strictly
    easier to earn. This test pins the amended reading so a future change is deliberate,
    which is the same job the previous version of it did for the literal reading."""
    assert LB.LEARNABILITY_BAR_SE == 1.96
    assert LB.assign_verdict("t", _iv(0, -0.1, 0.1), 0.0197, 0.010).learn_beats_null
    assert not LB.assign_verdict("t", _iv(0, -0.1, 0.1), 0.0190, 0.010).learn_beats_null
    # the case that used to clear it and no longer does -- votes_consensus's shape
    assert not LB.assign_verdict("t", _iv(0, -0.1, 0.1), 0.011, 0.010).learn_beats_null
    # the amendment is disclosed in the function's own docstring, not only in a spec file
    assert "AMENDED" in LB.assign_verdict.__doc__


def test_verdict_carries_everything_the_rule_consumed():
    """A verdict with no numbers attached cannot be audited, and FR-014b requires the
    uncertainty to be stated alongside it."""
    v = LB.assign_verdict("t", _iv(0.005, 0.004, 0.006), 0.02, 0.01)
    for field in ("ev_delta", "ev_lo", "ev_hi", "learn_delta", "learn_se"):
        assert getattr(v, field) is not None
    assert "0.005" in str(v) or "+0.005" in str(v)


# =============================================================================================
# T031 — synthetic triggers in each quadrant, and the empty-vs-itself control
# =============================================================================================

def test_empty_versus_ITSELF_is_indeterminate():
    """THE control T035 checks for, and the one case where the right answer is known a
    priori. A trigger compared with itself has an economic-value difference of exactly zero
    and a learnability difference of exactly zero, so it can clear NEITHER bar and must come
    out `indeterminate`. Any other result means the rule is reading something other than the
    difference -- a sign error, a reference mix-up, or an interval that excludes zero when
    the point estimate IS zero."""
    v = LB.assign_verdict("empty", _iv(0.0, 0.0, 0.0), 0.0, 0.0)
    assert v.verdict == LB.VERDICT_INDETERMINATE
    assert not v.ev_beats_null and not v.learn_beats_null


def test_empty_vs_itself_stays_indeterminate_even_with_a_degenerate_zero_width_interval():
    """A self-comparison's bootstrap interval collapses to [0, 0] (every resample gives the
    same two means). `lo > 0` must be False there -- a `>=` would turn the control into a
    false `selection_value`, which is exactly the failure the control exists to catch."""
    for se in (0.0, 1e-12, 0.01):
        v = LB.assign_verdict("empty", _iv(0.0, 0.0, 0.0), 0.0, se)
        assert v.verdict == LB.VERDICT_INDETERMINATE, f"se={se}"


@pytest.mark.parametrize("name,ev,learn,expected", [
    # a trigger with real selection value: EV clearly above the null
    ("selective", (+0.010, +0.009, +0.011), (+0.05, 0.005), LB.VERDICT_BEATS_NULL_FILLABLE),
    # one that only looks good to a model: no EV, strong learnability
    ("decorative", (-0.004, -0.005, -0.003), (+0.05, 0.005), LB.VERDICT_DECORATION),
    # one that is simply inert on both axes
    ("inert", (+0.0001, -0.0003, +0.0005), (+0.0007, 0.010), LB.VERDICT_INDETERMINATE),
    # negative on BOTH -- still indeterminate, never "selection value" by sign confusion
    # `worse_than_null` since the 2026-10-09 amendment: measured below the null AND not
    # learnable, which `indeterminate` used to file alongside the controls
    ("harmful", (-0.010, -0.011, -0.009), (-0.05, 0.005), LB.VERDICT_WORSE_THAN_NULL),
])
def test_synthetic_triggers_land_in_their_intended_quadrant(name, ev, learn, expected):
    """Constructed so each quadrant is exercised by a trigger whose intended verdict is
    known from how it was built, not from what the rule returns."""
    v = LB.assign_verdict(name, _iv(*ev), learn[0], learn[1])
    assert v.verdict == expected


def test_gbb_tight_is_withdrawn_from_adjudication():
    """Withdrawn 2026-10-09 (T035, user-approved): n=94 fillable non-overlapping trades, one
    LUNA-collapse week supplying 56.5% of summed profit from 22 of them, and the bootstrap
    resampling those 22 same-time cross-pair trades as independent blocks. A verdict asserts
    the interval is trustworthy; here it is not, so the row reports numbers and a reason."""
    assert "gbb_tight" in LB.DIAGNOSTIC_ONLY
    s = LB.economic_value(*_frames([0.1] * 5, [0] * 5), "empty")
    row = LB.assemble_row("gbb_tight", s, s, _iv(0.03, 0.02, 0.04), 0.1, 0.01, 0.1)
    assert row.verdict == LB.VERDICT_DIAGNOSTIC, "an EV-beating interval must not adjudicate it"


def test_controls_are_adjudicated_not_exempted():
    """A control that is not checked is not a control. `random_matched` and `gbb_stale` get
    real verdicts, and the expected one is `indeterminate`."""
    assert not (LB.CONTROL_TRIGGERS & LB.DIAGNOSTIC_ONLY)
    s = LB.economic_value(*_frames([0.1] * 5, [0] * 5), "empty")
    row = LB.assemble_row("random_matched", s, s, _iv(0.0003, -0.0003, 0.0009), 0.03, 0.01, 0.0)
    assert row.verdict == LB.VERDICT_INDETERMINATE


# =============================================================================================
# Concurrency-aware economic value (2026-10-09) — calibrated against a REAL arm
# =============================================================================================

def test_concurrency_cap_binds_and_takes_signals_in_time_order():
    """Earliest-first across the merged stream, capped at `max_open`. Five simultaneous
    signals with a cap of 2 must yield 2 taken."""
    T.register(T.Trigger("always", lambda d: pd.Series(True, index=d.index), ()))
    frames = [_frames([0.01] * 6, [5] * 6) for _ in range(5)]   # 5 pairs, long holds
    r = LB.concurrent_economic_value(frames, "always", max_open=2)
    assert r["n_taken"] < r["n_signals"], "the cap must actually bind"
    assert r["take_rate"] < 1.0


def test_an_unlimited_cap_takes_every_signal():
    T.register(T.Trigger("always", lambda d: pd.Series(True, index=d.index), ()))
    frames = [_frames([0.01] * 6, [0] * 6) for _ in range(3)]
    r = LB.concurrent_economic_value(frames, "always", max_open=10_000)
    assert r["take_rate"] == pytest.approx(1.0)


def test_a_position_blocks_only_until_it_closes():
    """`[entry, entry + hold]` occupancy. With a cap of 1 and hold 0, every bar is takeable;
    with hold 5, most are not."""
    T.register(T.Trigger("always", lambda d: pd.Series(True, index=d.index), ()))
    free = LB.concurrent_economic_value([_frames([0.01] * 10, [0] * 10)], "always", max_open=1)
    held = LB.concurrent_economic_value([_frames([0.01] * 10, [5] * 10)], "always", max_open=1)
    assert free["n_taken"] > held["n_taken"]


def test_CALIBRATION_against_the_breakout_vol_arm_is_recorded():
    """The number this function exists to reproduce, pinned as documentation.

    The harness ran `breakout_vol` for real (15 folds, 8,290 trades, -17.17pp/yr) and the
    live mean profit_ratio was -0.001512. Concurrency-aware EV over the UNFILTERED population
    gives -0.001548 -- within 3.6e-5. Over the FILLABLE population it gives +0.000545, close
    to the board's +0.000582 and of the OPPOSITE SIGN to live.

    That is the whole finding: the board's positive figure describes the fillable subset, and
    the live arm does not restrict to it -- `custom_stake_amount` CAPS the stake on an
    illiquid bar rather than rejecting the trade, so the position is opened anyway and
    `profit_ratio` (stake-independent) tracks the unfiltered population.

    Asserted as a docstring fact rather than recomputed, because recomputing needs the
    feather cache and the pooled arm; `scratchpad/calib.py` reproduces it.
    """
    doc = LB._select_concurrent.__doc__
    assert "-0.000547" in doc and "+0.000582" in doc and "-0.001512" in doc
