"""The trigger registry (T022, FR-010).

Every constraint is tested in the FAILING direction as well as the passing one. A guard
that cannot fail is worthless -- this feature already carries a permanent self-test that
plants a duplicate to prove the duplication detector fires (T005), and the same reasoning
applies here: a registry that accepts an outcome-reading trigger while its tests pass is
worse than no registry, because it certifies the thing it was built to prevent.
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

from research.outcomes import triggers as T  # noqa: E402
from research.outcomes.leakage import OUTCOME_COLUMN_POSITIVE  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Each test gets an empty registry, so ordering cannot couple them."""
    T.clear_registry()
    yield
    T.clear_registry()


def _frame(n: int = 50, pair: str = "SOL/USDT") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "pair": pair,
        "close": 100.0 + rng.normal(0, 1, n).cumsum(),
        "guard_metric": rng.uniform(0.0, 1.0, n),
        "bb_width": rng.uniform(0.0, 0.08, n),
        OUTCOME_COLUMN_POSITIVE: rng.integers(0, 2, n).astype(bool),
    })


def _always(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


# =============================================================================================
# Registration
# =============================================================================================

def test_registers_and_retrieves():
    t = T.register(T.Trigger("always", _always, ()))
    assert T.get("always") is t
    assert T.names() == ["always"]


def test_duplicate_name_raises_unless_replace_is_asked_for():
    """Silent replacement would change what every leaderboard row citing the name means."""
    T.register(T.Trigger("dup", _always, ()))
    with pytest.raises(T.TriggerError, match="already registered"):
        T.register(T.Trigger("dup", _always, ()))
    T.register(T.Trigger("dup", _always, (), description="second"), replace=True)
    assert T.get("dup").description == "second"


def test_unknown_name_raises_and_lists_what_exists():
    T.register(T.Trigger("known", _always, ()))
    with pytest.raises(T.TriggerError, match="known"):
        T.get("nope")


# =============================================================================================
# Constraint 1: no outcome column (FR-009). Both layers.
# =============================================================================================

def test_registration_rejects_a_trigger_declaring_an_outcome_column():
    """Layer 1. The label would be `outcome AND outcome` -- a perfect score meaning nothing."""
    with pytest.raises(T.TriggerError, match="may not read its own answer"):
        T.Trigger("circular", _always, (OUTCOME_COLUMN_POSITIVE,))


def test_evaluation_hides_outcome_columns_from_a_trigger_that_evaded_layer_one():
    """Layer 2, and the test that proves the guard can fail.

    This trigger declares nothing, so registration cannot catch it, and then reaches for
    the outcome column at evaluation time. It must raise rather than succeed."""
    def sneaky(df: pd.DataFrame) -> pd.Series:
        return df[OUTCOME_COLUMN_POSITIVE].astype(bool)

    T.register(T.Trigger("sneaky", sneaky, ()))
    df = _frame()
    assert OUTCOME_COLUMN_POSITIVE in df.columns, "the column must be present to be hidden"
    with pytest.raises(KeyError):
        T.evaluate("sneaky", df)


def test_the_same_trigger_works_once_the_outcome_column_is_not_what_it_reads():
    """Control for the test above: the failure is the outcome column, not the mechanism."""
    T.register(T.Trigger("fine", lambda d: d["close"] > 0, ("close",)))
    assert T.evaluate("fine", _frame()).all()


# =============================================================================================
# Constraint 2: no pair-specific behaviour (Constitution X). Both layers.
# =============================================================================================

def test_registration_rejects_a_trigger_declaring_the_pair_column():
    with pytest.raises(T.TriggerError, match="per-pair override"):
        T.Trigger("per_pair", _always, ("pair",))


def test_evaluation_is_INVARIANT_to_the_pair_label():
    """The measurement, not the argument. Identical bars under two different pair labels
    must give an identical mask -- this is what makes "not pair-specific" a fact about the
    registry rather than a claim in a docstring."""
    def pair_keyed(df: pd.DataFrame) -> pd.Series:
        # Tries to condition on the pair, defensively falling back so that the ONLY
        # reason the two runs can agree is that the column is genuinely unreachable.
        if "pair" in df.columns:
            return pd.Series(df["pair"].iloc[0] == "ZEC/USDT", index=df.index)
        return df["close"] > df["close"].median()

    T.register(T.Trigger("pair_keyed", pair_keyed, ("close",)))
    a = T.evaluate("pair_keyed", _frame(pair="ZEC/USDT"))
    b = T.evaluate("pair_keyed", _frame(pair="SOL/USDT"))
    pd.testing.assert_series_equal(a, b)
    assert not a.all() and a.any(), "fell through to the price branch, as it must"


# =============================================================================================
# required_columns asserted BEFORE evaluation (FR-010)
# =============================================================================================

def test_missing_required_column_raises_instead_of_firing_on_nothing():
    """The failure mode this exists to prevent: a trigger over an absent column fires on
    no bars, which is indistinguishable from a trigger that is genuinely rare."""
    T.register(T.Trigger("needs_rsi", lambda d: d["rsi"] > 70, ("rsi",)))
    with pytest.raises(T.TriggerError, match="fires on nothing"):
        T.evaluate("needs_rsi", _frame())


def test_a_withheld_column_is_named_as_withheld_not_merely_missing():
    """If a trigger asks for a column the registry deliberately hides, the error must say
    so -- otherwise the author hunts for a data problem that does not exist. (Reachable
    only by bypassing the dataclass check, as a direct `evaluate` call does.)"""
    t = T.Trigger.__new__(T.Trigger)
    object.__setattr__(t, "name", "withheld")
    object.__setattr__(t, "evaluate", _always)
    object.__setattr__(t, "required_columns", (OUTCOME_COLUMN_POSITIVE,))
    object.__setattr__(t, "source", T.SOURCE_NATIVE)
    object.__setattr__(t, "description", "")
    with pytest.raises(T.TriggerError, match="withheld from triggers"):
        T.evaluate(t, _frame())


# =============================================================================================
# Output validation
# =============================================================================================

def test_rejects_a_non_series_return():
    T.register(T.Trigger("arr", lambda d: np.ones(len(d), dtype=bool), ()))
    with pytest.raises(T.TriggerError, match="expected pd.Series"):
        T.evaluate("arr", _frame())


def test_rejects_a_misaligned_series():
    """A positionally-correct but differently-indexed mask silently misaligns against the
    outcome table, which is a wrong answer rather than an error."""
    T.register(T.Trigger("short", lambda d: pd.Series(True, index=d.index[:10]), ()))
    with pytest.raises(T.TriggerError, match="misaligned"):
        T.evaluate("short", _frame())


@pytest.mark.parametrize("mask", [
    pd.array([True, pd.NA], dtype="boolean"),        # nullable boolean
    np.array([True, None], dtype=object),            # object with None
], ids=["nullable_boolean", "object_None"])
def test_rejects_a_mask_that_still_carries_missingness(mask):
    """`NaN -> False` would turn a bar the trigger could not evaluate into 'did not
    fire'. Catchable whenever the missingness survives into the returned mask."""
    T.register(T.Trigger("nullable", lambda d: pd.Series(mask, index=d.index), ()))
    with pytest.raises(T.TriggerError, match="NaN"):
        T.evaluate("nullable", _frame(n=2))


def test_pandas_itself_swallows_warmup_NaN_so_the_guard_cannot_see_it():
    """THE LIMIT OF THE GUARD ABOVE, pinned so it stays discoverable.

    A rolling warm-up is the case the NaN check appears to protect against, and it does
    NOT: pandas evaluates a comparison against NaN to `False`, so the mask arrives clean
    `bool` dtype with its warm-up silently non-firing. Indistinguishable, here and
    downstream, from a trigger that genuinely did not fire.

    This is asserted rather than fixed because the fix belongs where the denominator
    lives -- `base_rate` in T027 -- not in output validation. If a future pandas makes
    such a comparison NaN-preserving, this test fails and points at the decision."""
    def warmup(df: pd.DataFrame) -> pd.Series:
        s = df["close"].rolling(10).mean()
        return s > s.shift(1)

    T.register(T.Trigger("warmup", warmup, ("close",)))
    out = T.evaluate("warmup", _frame())        # does NOT raise
    assert out.dtype == bool
    assert not out.iloc[:10].any(), "warm-up bars present as non-firing, not as missing"


def test_coerces_a_clean_integer_mask_to_bool():
    T.register(T.Trigger("ints", lambda d: (d["close"] > 0).astype(int), ("close",)))
    out = T.evaluate("ints", _frame())
    assert out.dtype == bool and out.all()


def test_a_trigger_cannot_mutate_the_callers_frame():
    def vandal(df: pd.DataFrame) -> pd.Series:
        df["close"] = -1.0
        return df["close"] < 0

    T.register(T.Trigger("vandal", vandal, ("close",)))
    df = _frame()
    before = df["close"].copy()
    T.evaluate("vandal", df)
    pd.testing.assert_series_equal(df["close"], before)


# =============================================================================================
# CLI (contracts/trigger-cli.md)
# =============================================================================================

def test_list_cli_reports_registered_triggers(capsys):
    T.register(T.Trigger("gbb_like", _always, ("guard_metric", "bb_width"), source="gbb"))
    assert T.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "gbb_like" in out and "guard_metric" in out and "gbb" in out


def test_list_cli_on_an_empty_registry_says_so_rather_than_printing_a_bare_header(capsys):
    assert T.main(["--list"]) == 0
    assert "no triggers registered" in capsys.readouterr().out


def test_cli_without_a_flag_is_an_error_not_a_silent_success(capsys):
    assert T.main([]) == 1
