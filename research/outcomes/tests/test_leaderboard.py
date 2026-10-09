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
    n = n or len(profit)
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
