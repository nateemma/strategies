"""Label formation (T024, FR-011) and the property that makes the leaderboard mean
anything (T026): two labels differ ONLY where their triggers differ.

That property is what the whole feature buys. Under the old scheme two labels differed in
both their entry condition and their forward-looking outcome test at once, so an arm's win
could not be attributed. Here it is a testable invariant, so it is tested -- including in
the direction that would break it.
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
from research.outcomes import labels as L  # noqa: E402
from research.outcomes import triggers as T  # noqa: E402
from research.outcomes.leakage import (  # noqa: E402
    OUTCOME_COLUMN_POSITIVE,
    OUTCOME_COLUMN_PROFIT,
    LeakageError,
)


@pytest.fixture(autouse=True)
def _registry():
    T.clear_registry()
    B.register_builtins()
    yield
    T.clear_registry()


N = 300


# Since 2026-10-09 a trigger on bar i pairs with outcome row i+1 (`labels.entry_mask`), so
# bar 0 is never an entry. Tests that compare a label against a raw trigger mask must
# compare against the SHIFTED mask; `_shift` is that, written out here rather than imported
# so these tests do not verify the helper against itself.
def _shift(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(mask, dtype=bool))
    out[1:] = np.asarray(mask, dtype=bool)[:-1]
    return out


def _features(n: int = N, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "close": 100.0 + rng.normal(0, 1, n).cumsum(),
        "guard_metric": rng.uniform(0.0, 1.0, n),
        "bb_width": rng.uniform(0.0, 0.08, n),
        "rvol": rng.uniform(0.0, 3.0, n),
    })


def _outcomes(features: pd.DataFrame, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(features)
    return pd.DataFrame({
        "date": features["date"].to_numpy(),
        OUTCOME_COLUMN_POSITIVE: rng.integers(0, 2, n).astype(bool),
        OUTCOME_COLUMN_PROFIT: rng.normal(0, 0.02, n),
    })


# =============================================================================================
# The conjunction itself
# =============================================================================================

def test_label_is_the_conjunction_of_trigger_and_positive_outcome():
    f, o = _features(), None
    o = _outcomes(f)
    mask = L.trigger_label_mask(f, o, "gbb")
    expected = _shift(T.evaluate("gbb", f).to_numpy()) & o[OUTCOME_COLUMN_POSITIVE].to_numpy()
    np.testing.assert_array_equal(mask.to_numpy(), expected)


def test_the_empty_trigger_reduces_the_label_to_the_outcome_alone():
    """`empty` fires everywhere, so its label IS `__outcome_positive`. That is what makes
    it the verdict rule's null: the forward_mae case is the no-trigger case."""
    f = _features()
    o = _outcomes(f)
    # `empty` fires everywhere, so its ENTRY mask is every bar except the first.
    expected = _shift(T.evaluate("empty", f).to_numpy()) & o[OUTCOME_COLUMN_POSITIVE].to_numpy()
    np.testing.assert_array_equal(L.trigger_label_mask(f, o, "empty").to_numpy(), expected)


def test_a_label_never_exceeds_its_trigger():
    """A conjunction can only narrow. If a label ever fired where its trigger did not,
    the outcome would be acting as an entry condition of its own."""
    f = _features()
    o = _outcomes(f)
    for name in T.names():
        fired = _shift(T.evaluate(name, f).to_numpy())
        labelled = L.trigger_label_mask(f, o, name).to_numpy()
        assert not (labelled & ~fired).any(), name


def test_there_is_no_label_source_parameter_to_reintroduce_a_per_label_outcome():
    """FR-011 structurally, not by convention. An optional outcome argument is how the old
    scheme -- every label carrying its own forward test -- would grow back."""
    import inspect

    for fn in (L.trigger_label_mask, L.build_trigger_label, L.label_rate):
        params = set(inspect.signature(fn).parameters)
        assert params == {"features", "outcomes", "trigger"}, (fn.__name__, params)


# =============================================================================================
# T026's property: labels differ only where triggers differ
# =============================================================================================

def test_two_labels_differ_ONLY_where_their_triggers_differ():
    """The invariant the leaderboard rests on. With one shared outcome, any difference
    between two labels must be attributable to the triggers -- so a difference anywhere
    the triggers AGREE would mean an outcome definition leaked in."""
    f = _features()
    o = _outcomes(f)
    for a, b in (("gbb", "empty"), ("gbb", "geometry"), ("geometry", "empty")):
        ta, tb = _shift(T.evaluate(a, f).to_numpy()), _shift(T.evaluate(b, f).to_numpy())
        la = L.trigger_label_mask(f, o, a).to_numpy()
        lb = L.trigger_label_mask(f, o, b).to_numpy()
        agree = ta == tb
        np.testing.assert_array_equal(
            la[agree], lb[agree],
            err_msg=f"{a} vs {b}: labels differ where the triggers agree",
        )
        assert (la != lb).any(), f"{a} vs {b}: triggers differ but labels never do"


def test_the_invariant_FAILS_if_a_second_outcome_definition_is_introduced():
    """Proof the test above can fail. Labels built against two DIFFERENT outcome columns
    differ even where the triggers agree -- which is precisely the old scheme, and what
    this design removes."""
    f = _features()
    o = _outcomes(f)
    alt = o.copy()
    alt[OUTCOME_COLUMN_POSITIVE] = o[OUTCOME_COLUMN_PROFIT] > 0.0   # a rival definition

    ta = _shift(T.evaluate("gbb", f).to_numpy())
    la = L.trigger_label_mask(f, o, "gbb").to_numpy()
    lb = L.trigger_label_mask(f, alt, "gbb").to_numpy()
    agree = np.ones_like(ta, dtype=bool)        # same trigger, so they agree everywhere
    assert (la[agree] != lb[agree]).any(), (
        "two rival outcome definitions produced identical labels; the fixture is too weak "
        "to demonstrate what the invariant forbids"
    )


# =============================================================================================
# Alignment — the expensive bug this refuses to build
# =============================================================================================

def test_a_length_mismatch_is_refused():
    f = _features()
    with pytest.raises(L.AlignmentError, match="same bars"):
        L.trigger_label_mask(f, _outcomes(f).iloc[:-1], "gbb")


def test_a_ONE_BAR_OFFSET_is_refused_even_though_the_lengths_match():
    """THE test. A one-bar offset reads as a label that predicts the future slightly
    better than it should -- indistinguishable from an edge, and a class of error this
    repository has already paid for (an oracle arm entering two bars early, with a whole
    theory built on the bug before the instrument was audited). Row counts are identical
    here, so only the date comparison can catch it."""
    f = _features()
    o = _outcomes(f)
    shifted = o.copy()
    shifted["date"] = shifted["date"] + pd.Timedelta(minutes=15)
    assert len(shifted) == len(f)
    with pytest.raises(L.AlignmentError, match="one-bar offset"):
        L.trigger_label_mask(f, shifted, "gbb")


def test_alignment_passes_on_matching_dates():
    f = _features()
    L.assert_aligned(f, _outcomes(f))          # must not raise


def test_dateless_frames_fall_back_to_the_length_check_only():
    """Documented downgrade, not a silent one: with no `date` on either side the length
    check is all that exists, and the docstring says so."""
    f = _features().drop(columns=["date"])
    o = _outcomes(_features()).drop(columns=["date"])
    L.assert_aligned(f, o)
    assert "weaker check" in L.assert_aligned.__doc__


# =============================================================================================
# The leakage boundary is reused, not reimplemented
# =============================================================================================

def test_build_trigger_label_attaches_one_label_column_and_no_outcome_columns():
    f = _features()
    out = L.build_trigger_label(f, _outcomes(f), "gbb")
    assert L.LABEL_COLUMN in out.columns
    assert not [c for c in out.columns if c.startswith("__outcome_")]
    assert list(f.columns) == [c for c in out.columns if c != L.LABEL_COLUMN]


def test_passing_the_outcome_bearing_frame_as_features_is_caught():
    """The FR-009 failure the boundary exists for: a caller that hands the table in as
    "features" would train a model on its own answer."""
    f = _features()
    o = _outcomes(f)
    contaminated = f.join(o[[OUTCOME_COLUMN_POSITIVE]])
    with pytest.raises(LeakageError, match="label builder input"):
        L.build_trigger_label(contaminated, o, "gbb")


def test_missing_outcome_column_is_a_clear_error_not_a_KeyError():
    f = _features()
    with pytest.raises(LeakageError, match="Build the table first"):
        L.trigger_label_mask(f, f[["date"]], "gbb")


def test_a_trigger_still_cannot_reach_an_outcome_column_through_this_path():
    """The registry's guarantee must survive the extra indirection: the trigger is
    evaluated against `features` only, never against anything carrying the outcome."""
    def sneaky(df: pd.DataFrame) -> pd.Series:
        return df[OUTCOME_COLUMN_POSITIVE].astype(bool)

    T.register(T.Trigger("sneaky", sneaky, ()))
    f = _features()
    with pytest.raises(KeyError):
        L.trigger_label_mask(f, _outcomes(f), "sneaky")


# =============================================================================================
# label_rate
# =============================================================================================

def test_label_rate_matches_the_mask_mean_and_is_bounded_by_the_base_rate():
    f = _features()
    o = _outcomes(f)
    for name in T.names():
        rate = L.label_rate(f, o, name)
        assert rate == pytest.approx(L.trigger_label_mask(f, o, name).mean())
        assert rate <= _shift(T.evaluate(name, f).to_numpy()).mean() + 1e-12, name


# =============================================================================================
# THE PAIRING (2026-10-09) — the defect a gate review caught
# =============================================================================================

def test_entry_mask_lags_the_trigger_by_exactly_one_bar():
    """A trigger reads its bar's CLOSE, so the earliest price it can transact at is the NEXT
    bar's open — which is the outcome table's row i+1. Pairing directly booked the signal
    bar's own open-to-close move as profit: 91-94% of the breakout family's apparent
    economic value, and a sign flip for geometry."""
    f = _features(n=10)
    o = _outcomes(f)
    T.register(T.Trigger("bar3", lambda d: pd.Series(
        np.arange(len(d)) == 3, index=d.index), ()))
    m = L.entry_mask(f, o, "bar3").to_numpy()
    assert m.sum() == 1
    assert m[4], "a signal on bar 3 must enter on bar 4"
    assert not m[3], "entering on the signal bar is the leak"


def test_the_first_bar_is_never_an_entry():
    """Nothing fired before it, so there is no signal it could be acting on."""
    f = _features(n=50)
    o = _outcomes(f)
    for name in T.names():
        assert not L.entry_mask(f, o, name).to_numpy()[0], name


def test_a_signal_on_the_last_bar_is_dropped_rather_than_wrapped():
    """There is no next bar to enter on. Wrapping to index 0 would pair a late signal with
    the start of the series — the same class of error as `labels_bands`' np.roll."""
    f = _features(n=10)
    o = _outcomes(f)
    T.register(T.Trigger("last", lambda d: pd.Series(
        np.arange(len(d)) == len(d) - 1, index=d.index), ()))
    assert L.entry_mask(f, o, "last").to_numpy().sum() == 0


def test_a_signal_before_a_DATA_GAP_is_dropped():
    """A signal on bar i is only actionable at open[i+1] if bar i+1 is the next bar in time.
    Across a gap it is not, and pairing anyway would book an entry at a price hours or days
    later as though it were the next bar."""
    f = _features(n=10)
    dates = list(f["date"])
    dates[5:] = [d + pd.Timedelta(hours=6) for d in dates[5:]]   # gap after bar 4
    f = f.assign(date=dates)
    o = _outcomes(f)
    T.register(T.Trigger("bar4", lambda d: pd.Series(
        np.arange(len(d)) == 4, index=d.index), ()))
    assert L.entry_mask(f, o, "bar4").to_numpy().sum() == 0, (
        "a signal immediately before a data gap must be dropped"
    )


def test_empty_is_the_CONTROL_for_the_shift():
    """`empty` fires on every bar, so shifting it changes only the first bar. On real data
    its economic value was unchanged to 2e-6 by the fix, which is what proved the shift
    itself introduces nothing and the effect it removed was real."""
    f = _features(n=100)
    o = _outcomes(f)
    fired = T.evaluate("empty", f).to_numpy()
    entry = L.entry_mask(f, o, "empty").to_numpy()
    assert fired.all()
    assert entry.sum() == len(f) - 1
    np.testing.assert_array_equal(entry[1:], fired[1:])
