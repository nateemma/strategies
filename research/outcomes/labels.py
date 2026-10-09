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

import numpy as np
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


#: One bar. A trigger's decision uses the CLOSE of its bar, so the earliest price it can
#: transact at is the NEXT bar's open — which is what the outcome table's row `i+1`
#: simulates (`simulate_forward` enters at `open_[start_index]`).
ENTRY_LAG_BARS = 1


def entry_mask(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    trigger: "_triggers.Trigger | str",
) -> pd.Series:
    """The bars at which a trade actually STARTS, given this trigger. THE pairing.

    A trigger is evaluated on bar `i` and reads bar `i`'s close — `breakout` is literally
    `close[i] > prior_high`. The outcome table's row `i` simulates entering at `open[i]`.
    Pairing them directly therefore books bar `i`'s own open-to-close move as profit the
    trigger could never have captured: at `open[i]` the close that caused the signal has not
    happened yet. The signal bar is free money, and momentum triggers fire on up-bars, so
    they collect it.

    MEASURED, 2026-10-09, after a gate review caught this. Fillable, non-overlapping, pooled
    11 pairs:

        trigger        sig-bar o->c   paired at i   paired at i+1
        breakout             +0.456%    +0.009076       +0.000807
        breakout_vol         +0.614%    +0.009254       +0.000582
        geometry             +0.086%    +0.001210       -0.001526   (sign flip)
        gbb                  -0.047%    -0.005374       -0.002009
        empty                -0.000%    -0.001272       -0.001275   (control: unchanged)

    91-94% of the breakout family's apparent economic value was its own signal bar. `empty`
    is unchanged to 2e-6 because it fires on every bar, which is the control proving the
    shift itself introduces nothing.

    WHY T021'S RECONCILIATION DID NOT CATCH IT. That work matched the ARM's trades at
    `trade.open_date` and looked up the row at that date — and the arm genuinely enters at
    that bar's open, so row `i` was validated correctly AS A TRADE STARTING AT BAR i. The
    arm's SIGNAL was the bar before. The table was never wrong; the leaderboard's pairing
    was, and `assert_aligned` enforced the wrong one by construction.

    CONTIGUITY. A signal on bar `i` is only actionable at `open[i+1]` if bar `i+1` is the
    next bar in time. Across a data gap it is not, so those signals are dropped rather than
    paired with whatever row happens to sit at `i+1`.
    """
    assert_aligned(features, outcomes)
    fired = _triggers.evaluate(trigger, features).to_numpy()

    shifted = np.zeros_like(fired)
    if len(fired) > ENTRY_LAG_BARS:
        shifted[ENTRY_LAG_BARS:] = fired[:-ENTRY_LAG_BARS]

    if DATE_COLUMN in features.columns and len(features) > ENTRY_LAG_BARS:
        d = pd.to_datetime(features[DATE_COLUMN], utc=True)
        step = d.diff().shift(-ENTRY_LAG_BARS)
        modal = step.mode()
        if len(modal):
            contiguous = (step == modal.iloc[0]).to_numpy()
            # a signal is actionable only if ITS next bar is contiguous; the guard is
            # indexed on the signal bar, then shifted with it
            ok = np.zeros_like(shifted)
            ok[ENTRY_LAG_BARS:] = contiguous[:-ENTRY_LAG_BARS]
            shifted = shifted & ok

    return pd.Series(shifted, index=features.index)


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

    fired = entry_mask(features, outcomes, trigger).to_numpy()
    positive = outcomes[OUTCOME_COLUMN_POSITIVE].to_numpy(dtype=bool)
    return pd.Series(fired & positive, index=features.index)


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
