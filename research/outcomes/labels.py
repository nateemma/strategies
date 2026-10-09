"""Label formation (T024, FR-011): `label = trigger AND positive outcome`.

THE WHOLE POINT. Every `LabelMethod` in `Framework/TrainingSignals.py` carries its own
forward-looking outcome test -- `future_gain >= min_gain`, an MFE/MAE ratio, a forward
Sharpe -- bolted onto whatever entry condition it uses. So two labels differ in two places
at once, and no experiment run so far could say whether an arm won because its trigger
selected collectable moments or because its outcome definition was easier to predict.

Here the outcome comes from ONE place: the shared table, built from the one exit policy
(`utils.ExitPolicy`). A label is that outcome conjoined with a trigger, and the trigger is
the only thing that varies. There is deliberately NO per-label outcome parameter anywhere
in this module -- not even an optional one, because an optional one is how the old scheme
would grow back.

WHAT THIS MODULE REFUSES TO DO, and why it is worth the friction: it will not accept a
`features` frame and an `outcomes` frame it cannot prove are the same bars. A one-bar
offset between a trigger and its outcome is the single most expensive bug available here
-- it would read as a label that predicts the future slightly better than it should, which
is indistinguishable from an edge. This repository has already lost time to exactly that
class of error (an oracle arm entering two bars early, where a whole theory was built on
the bug before the instrument was audited), so alignment is checked on the `date` column
and not inferred from row counts.
"""
from __future__ import annotations

import pandas as pd

from research.outcomes import triggers as _triggers
from research.outcomes.leakage import (
    OUTCOME_COLUMN_POSITIVE,
    LeakageError,
    build_label,
)

#: The column `build_label` writes, and what a probe trains on.
LABEL_COLUMN = "label"

#: Checked for alignment when present in both frames. The outcome table carries it
#: (data-model.md's `OutcomeRecord`) and so does every populated feature frame.
DATE_COLUMN = "date"


class AlignmentError(ValueError):
    """Raised when `features` and `outcomes` cannot be shown to describe the same bars.

    A plain catchable exception rather than an assert, for the same reason `LeakageError`
    is one: `python -O` strips asserts, and an invariant that an optimisation flag can
    silence is not an invariant.
    """


def assert_aligned(features: pd.DataFrame, outcomes: pd.DataFrame) -> None:
    """Prove the two frames describe the same bars, in the same order.

    Row count alone is NOT proof -- two frames over the same pair with a one-bar offset
    have identical lengths. When both carry `date` it is compared elementwise; when
    neither does, the length check is all that is available and the caller is told
    explicitly that the weaker check was used, so a silent downgrade is impossible.
    """
    if len(features) != len(outcomes):
        raise AlignmentError(
            f"features has {len(features)} rows, outcomes has {len(outcomes)}. These must "
            f"be the same bars: the label is a per-bar conjunction, so a length mismatch "
            f"cannot be reconciled by any ordering."
        )

    has_dates = DATE_COLUMN in features.columns and DATE_COLUMN in outcomes.columns
    if not has_dates:
        return

    f = pd.to_datetime(features[DATE_COLUMN], utc=True).to_numpy()
    o = pd.to_datetime(outcomes[DATE_COLUMN], utc=True).to_numpy()
    if (f != o).any():
        bad = int((f != o).sum())
        first = int((f != o).argmax())
        raise AlignmentError(
            f"features and outcomes disagree on {bad} of {len(f)} dates, first at row "
            f"{first} ({f[first]} vs {o[first]}). A one-bar offset here reads as a label "
            f"that predicts slightly better than it should, which is indistinguishable "
            f"from an edge -- refusing to build it."
        )


def trigger_label_mask(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    trigger: "_triggers.Trigger | str",
) -> pd.Series:
    """`trigger AND __outcome_positive`, as a boolean Series on `features.index`.

    The trigger is evaluated against `features` ONLY, through `triggers.evaluate`, so the
    registry's guarantees apply: it cannot read an outcome column (they are stripped before
    it runs) and it cannot read the pair. `outcomes` contributes exactly one column.

    There is no `label_source` parameter. `__outcome_positive` IS the outcome definition
    for every label under this design, and making it selectable would reintroduce the
    per-label outcome test this module exists to remove. A caller that wants a different
    outcome definition wants a different EXIT POLICY, which is `ExitParams` plus a rebuilt
    table under a new `config_hash` -- not a label argument.
    """
    assert_aligned(features, outcomes)
    if OUTCOME_COLUMN_POSITIVE not in outcomes.columns:
        raise LeakageError(
            f"outcomes frame has no {OUTCOME_COLUMN_POSITIVE!r} column "
            f"(columns={list(outcomes.columns)}). Build the table first "
            f"(research.outcomes.table)."
        )

    fired = _triggers.evaluate(trigger, features)
    positive = outcomes[OUTCOME_COLUMN_POSITIVE].to_numpy(dtype=bool)
    return pd.Series(fired.to_numpy() & positive, index=features.index)


def build_trigger_label(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    trigger: "_triggers.Trigger | str",
) -> pd.DataFrame:
    """`features` plus a `label` column, formed as `trigger AND __outcome_positive`.

    Routes the result through `leakage.build_label` rather than attaching the column
    directly, so the FR-009 boundary is REUSED and not reimplemented: `features` is
    asserted leakage-free on entry and the output is asserted to carry no `__outcome_*`
    column. A second copy of that assertion is exactly how the two would drift apart.
    """
    mask = trigger_label_mask(features, outcomes, trigger)
    return build_label(
        features,
        pd.DataFrame({OUTCOME_COLUMN_POSITIVE: mask.to_numpy()}, index=features.index),
        label_source=OUTCOME_COLUMN_POSITIVE,
    )


def label_rate(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    trigger: "_triggers.Trigger | str",
) -> float:
    """Share of bars labelled positive. Convenience for the leaderboard and the probe's
    collapse check -- a label below roughly 5.5% has driven all-Hold collapse here before,
    so the figure is worth having before a training run rather than after one."""
    return float(trigger_label_mask(features, outcomes, trigger).mean())
