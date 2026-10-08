"""The leakage guard (feature 021, FR-009) — written BEFORE the table builder that
produces the columns it guards (tasks.md T013, strictly ordered ahead of T014).

Two layers, both tested in `research/outcomes/tests/test_leakage.py`:

1. **Reserved prefix.** Every outcome column is named `__outcome_*`
   (`OUTCOME_COLUMN_NAMES` below, taken verbatim from data-model.md's `OutcomeRecord`
   table). Model input is built from `Framework.FeatureNormalizer.include_list`, a
   fixed, hand-maintained list that cannot name a column that does not exist in it --
   so a prefixed column cannot be selected by accident. `test_leakage.py` pins that
   disjointness directly, ahead of `table.py` even existing, since the column names
   are defined here rather than there.

2. **A raising assertion at the label-builder boundary.** `assert_no_outcome_columns`
   raises `LeakageError` (never warns) if any `__outcome_*` column survives into a
   frame about to be handed onward; `build_label` is the boundary that calls it --
   the function Phase 4's trigger-based label formation (`trigger AND
   __outcome_positive`, FR-011) will route through once it exists. It exists now,
   ahead of the table builder, purely so the guard has a real boundary to prove
   itself against.

A convention alone would not be "structurally prevented" (FR-009) -- the prefix makes
accidental selection impossible, the assertion makes deliberate/buggy misuse loud.
"""
from __future__ import annotations

import pandas as pd

#: Load-bearing, not cosmetic (data-model.md, contracts/outcome-table.md): this is how
#: layer 1 of the guard works, and `FeatureNormalizer.include_list` never names a
#: column with this prefix.
OUTCOME_PREFIX = "__outcome_"

#: The seven `OutcomeRecord` fields, verbatim from data-model.md. Defined HERE, ahead
#: of `table.py` (T014), so the layer-1 test below can pin the naming convention
#: before anything produces these columns -- and so `table.py` names its columns from
#: this single list rather than restating the strings.
OUTCOME_COLUMN_PROFIT = f"{OUTCOME_PREFIX}profit"
OUTCOME_COLUMN_HOLD_BARS = f"{OUTCOME_PREFIX}hold_bars"
OUTCOME_COLUMN_CLOSE_REASON = f"{OUTCOME_PREFIX}close_reason"
OUTCOME_COLUMN_POSITIVE = f"{OUTCOME_PREFIX}positive"
OUTCOME_COLUMN_FILLABLE = f"{OUTCOME_PREFIX}fillable"
OUTCOME_COLUMN_FILLABLE_FRACTION = f"{OUTCOME_PREFIX}fillable_fraction"
OUTCOME_COLUMN_GAPFILL_SHARE = f"{OUTCOME_PREFIX}gapfill_share"

OUTCOME_COLUMN_NAMES: frozenset[str] = frozenset({
    OUTCOME_COLUMN_PROFIT,
    OUTCOME_COLUMN_HOLD_BARS,
    OUTCOME_COLUMN_CLOSE_REASON,
    OUTCOME_COLUMN_POSITIVE,
    OUTCOME_COLUMN_FILLABLE,
    OUTCOME_COLUMN_FILLABLE_FRACTION,
    OUTCOME_COLUMN_GAPFILL_SHARE,
})


class LeakageError(ValueError):
    """Raised when a `__outcome_*` column would otherwise cross the label-builder
    boundary. Deliberately a plain, catchable exception type (not a bare assert),
    so a caller cannot accidentally silence it with `python -O`."""


def outcome_columns(df: pd.DataFrame) -> list[str]:
    """Every column of `df` carrying the reserved `__outcome_` prefix."""
    return [c for c in df.columns if c.startswith(OUTCOME_PREFIX)]


def assert_no_outcome_columns(df: pd.DataFrame, *, boundary: str) -> None:
    """Raise `LeakageError` if `df` carries any `__outcome_*` column.

    `boundary` names the call site in the error message (e.g. "label builder
    output") so a failure points at where the leak was caught, not just that one
    occurred.
    """
    leaked = outcome_columns(df)
    if leaked:
        raise LeakageError(
            f"{boundary}: outcome column(s) {leaked} must not cross this boundary "
            f"(FR-009) -- the '{OUTCOME_PREFIX}' prefix exists so this is structurally "
            "prevented, not merely a convention."
        )


def build_label(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    label_source: str = OUTCOME_COLUMN_POSITIVE,
) -> pd.DataFrame:
    """The label-builder boundary (FR-009 layer 2, FR-011's eventual home).

    Consumes exactly one outcome column (`label_source`, `__outcome_positive` by the
    data model) out of `outcomes` to form the `label` column, attaches it to a copy
    of `features`, and asserts -- raising, not warning -- that nothing in the result
    still carries the reserved prefix.

    `features` is asserted leakage-free on entry too: a caller that accidentally
    passes the outcome-bearing frame (or a frame some other outcome column leaked
    into) as "features" is exactly the FR-009 failure this guard exists to catch
    loudly, before it ever reaches a model.

    Phase 4 (`research/outcomes/triggers.py`, not yet built) will call this with
    `label_source="__outcome_positive"` and a trigger-masked frame; nothing about
    this function is specific to that caller.
    """
    assert_no_outcome_columns(features, boundary="label builder input (features)")
    if label_source not in outcomes.columns:
        raise LeakageError(
            f"label builder: label_source={label_source!r} not found in outcomes "
            f"frame (columns={list(outcomes.columns)})"
        )
    result = features.copy()
    result["label"] = outcomes[label_source].to_numpy()
    assert_no_outcome_columns(result, boundary="label builder output")
    return result
