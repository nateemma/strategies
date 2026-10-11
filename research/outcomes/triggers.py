"""The trigger registry (T022, FR-010) — a trigger carries NO outcome test.

Feature 021's whole design is that a label is `trigger AND positive outcome`, where the
outcome comes from one shared exit policy, so the ONLY thing that varies between labels is
the trigger. That makes labels comparable for the first time: the evidence this feature
rests on could not distinguish a trigger that selects collectable moments from one that
hands the model an easy target, because every label carried its own outcome definition
baked in.

This module holds the triggers and nothing else. It deliberately cannot form a label --
that is `leakage.build_label`'s job, behind the leakage boundary (FR-011).

TWO CONSTRAINTS, ENFORCED STRUCTURALLY RATHER THAN BY CONVENTION
----------------------------------------------------------------
Both are checked at REGISTRATION (so a bad trigger cannot enter the registry) and again by
dropping the offending columns before `evaluate` is ever called (so a trigger that evades
the static check still fails loudly with a `KeyError` instead of returning a wrong answer).
This repository has been burned by convention-only invariants -- duplicated loss logic
where the wiring checks passed while the term never ran -- which is why neither layer is
trusted alone.

1. **No outcome column** (FR-009). A trigger that reads `__outcome_positive` would be
   circular: the label would be `outcome AND outcome`, scoring perfectly and meaning
   nothing. `required_columns` is rejected if it names one, and `evaluate` is handed a
   frame with every `__outcome_*` column REMOVED.

2. **No pair-specific behaviour** (Constitution X). A trigger conditioned on the pair is a
   per-pair override wearing a trigger's clothes, and this repository's edge has already
   been shown to concentrate in one high-volatility name -- a pair-keyed trigger would
   launder that concentration into something that looks structural. `pair` is rejected in
   `required_columns` and the column is REMOVED before evaluation.

   `PAIR_INVARIANCE_NOTE`: dropping the column makes the pair unreadable, which is the
   guarantee that matters. `test_triggers.py` additionally evaluates the same OHLCV under
   two different pair labels and asserts identical output, so the invariance is measured
   and not merely argued.

WHAT A TRIGGER MAY NOT DO, that is NOT enforced here: look forward. A trigger is evaluated
on the feature frame, whose columns are produced by `DataframePopulator` and are
backward-looking by construction. The outcome table is where lookahead legitimately lives
(it is label construction, not a feature), and it reaches a label only through
`build_label`. A trigger that computed its own forward window would be a new outcome
definition, which is the thing this design supersedes -- see T025's audit of the
`LabelMethod` members that do exactly that.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

from research.outcomes.leakage import OUTCOME_PREFIX, outcome_columns

#: Rejected in `required_columns` and dropped before evaluation (Constitution X).
PAIR_COLUMN = "pair"

#: `source` value for a trigger that is not a port of an existing `LabelMethod`.
SOURCE_NATIVE = "native"


class TriggerError(ValueError):
    """Raised when a trigger violates a registry constraint.

    A plain catchable exception rather than a bare `assert`, for the same reason
    `LeakageError` is: `python -O` strips asserts, and an invariant that can be
    silenced by an optimisation flag is not an invariant.
    """


@dataclass(frozen=True)
class Trigger:
    """A boolean condition over bars (data-model.md's `Trigger`).

    `evaluate` receives a dataframe with the outcome and `pair` columns already removed
    and must return a boolean Series aligned to its index. It must be a pure function of
    that frame: the registry calls it more than once (the leaderboard evaluates every
    trigger per pair) and nothing re-seeds it between calls.
    """

    name: str
    evaluate: Callable[[pd.DataFrame], pd.Series]
    required_columns: tuple[str, ...]
    source: str = SOURCE_NATIVE
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise TriggerError("a trigger needs a name")
        if not callable(self.evaluate):
            raise TriggerError(f"{self.name}: evaluate must be callable")

        bad_outcome = sorted(
            c for c in self.required_columns if c.startswith(OUTCOME_PREFIX)
        )
        if bad_outcome:
            raise TriggerError(
                f"{self.name}: required_columns names outcome column(s) {bad_outcome}. "
                f"A trigger may not read its own answer -- the label would be "
                f"'outcome AND outcome', which scores perfectly and means nothing "
                f"(FR-009/FR-010)."
            )
        if PAIR_COLUMN in self.required_columns:
            raise TriggerError(
                f"{self.name}: required_columns names {PAIR_COLUMN!r}. A trigger "
                f"conditioned on the pair is a per-pair override (Constitution X); this "
                f"repository's edge already concentrates in one high-volatility name, and "
                f"a pair-keyed trigger would launder that into something that looks "
                f"structural."
            )


#: name -> Trigger. Module-level and mutated by `register`, matching how
#: `ClassifierTypeMLX` and the `LabelMethod` dispatch are reached elsewhere in the repo.
_REGISTRY: dict[str, Trigger] = {}


def register(trigger: Trigger, *, replace: bool = False) -> Trigger:
    """Add `trigger` to the registry.

    Re-registering a name raises unless `replace=True`. Silent replacement is how a
    rename or a duplicated definition would quietly change every leaderboard row that
    cites the name, so it has to be asked for.
    """
    if trigger.name in _REGISTRY and not replace:
        raise TriggerError(
            f"{trigger.name!r} is already registered (from {_REGISTRY[trigger.name].source}). "
            f"Pass replace=True only if you mean to redefine it -- a silent replacement "
            f"changes what every leaderboard row carrying this name means."
        )
    _REGISTRY[trigger.name] = trigger
    return trigger


def get(name: str) -> Trigger:
    if name not in _REGISTRY:
        raise TriggerError(
            f"no trigger {name!r}; registered: {sorted(_REGISTRY) or '(none)'}"
        )
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)


def all_triggers() -> list[Trigger]:
    return [_REGISTRY[n] for n in names()]


def clear_registry() -> None:
    """Empty the registry. For tests only -- never call this from a pipeline."""
    _REGISTRY.clear()


def ensure_builtins() -> None:
    """Register the FR-013 members if they are not already present.

    Imported lazily, inside the function, so `builtin_triggers` can import this module at
    its top level without a cycle -- the registry must not depend on its own content.
    Idempotent, so a caller that has already registered them (or deliberately replaced
    one) is left alone.
    """
    from research.outcomes.builtin_triggers import (
        register_audit_triggers,
        register_builtins,
        register_margin_triggers,
    )

    required = {"empty", "gbb", "geometry"}
    audited = {"adx", "breakout", "breakout_gbb"}
    margin = {"gbb_tight"}
    remaining = {"breakout_vol", "breakout_squeeze", "breakout_bb", "votes_tech",
                 "votes_ind2", "votes_ind3", "votes_ind4", "votes_consensus",
                 "trends", "bands"}
    if not required <= set(_REGISTRY):
        register_builtins(replace=True)
    if not audited <= set(_REGISTRY):
        register_audit_triggers(replace=True)
    if not margin <= set(_REGISTRY):
        register_margin_triggers(replace=True)
    if not {"random_matched", "gbb_stale"} <= set(_REGISTRY):
        from research.outcomes.builtin_triggers import register_controls
        register_controls(replace=True)
    if not remaining <= set(_REGISTRY):
        from research.outcomes.audit_triggers import register_remaining_audit_triggers
        register_remaining_audit_triggers(replace=True)
    if not {"breakout_gm", "breakout_vol_gm", "breakout_squeeze_gm"} <= set(_REGISTRY):
        from research.outcomes.audit_triggers import register_momentum_guard_triggers
        register_momentum_guard_triggers(replace=True)


def _visible_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The frame a trigger is actually allowed to see.

    Layer 2 of both constraints: `__outcome_*` and `pair` are REMOVED, so a trigger that
    slipped past the registration check fails with a `KeyError` naming the column rather
    than silently returning a circular or pair-keyed answer. Returns a shallow copy, so a
    trigger that mutates what it is given cannot corrupt the caller's frame.
    """
    drop = outcome_columns(df) + ([PAIR_COLUMN] if PAIR_COLUMN in df.columns else [])
    return df.drop(columns=drop) if drop else df.copy()


def evaluate(trigger: Trigger | str, df: pd.DataFrame) -> pd.Series:
    """Evaluate `trigger` over `df`, enforcing the registry's guarantees.

    Asserts `required_columns` are present BEFORE evaluation (FR-010): a missing feature
    column otherwise surfaces as a trigger that quietly fires on nothing, which is
    indistinguishable from a trigger that is genuinely rare -- and this repository has
    already recorded that failure mode ("a broken probe reports 0, same as a real null").

    Returns a boolean Series on `df.index`, so a caller can align it to the outcome table
    without re-deriving the index.
    """
    trig = get(trigger) if isinstance(trigger, str) else trigger
    visible = _visible_frame(df)

    missing = [c for c in trig.required_columns if c not in visible.columns]
    if missing:
        hidden = [c for c in missing if c in df.columns]
        extra = (
            f" (present in the input but withheld from triggers: {hidden})" if hidden else ""
        )
        raise TriggerError(
            f"{trig.name}: required column(s) {missing} absent{extra}. Refusing to "
            f"evaluate: a trigger over a missing column fires on nothing, which looks "
            f"exactly like a trigger that is genuinely rare."
        )

    out = trig.evaluate(visible)
    return _validate_output(trig, out, visible.index)


def _validate_output(trigger: Trigger, out: object, index: pd.Index) -> pd.Series:
    """A trigger must return a clean boolean mask, not something mask-shaped.

    NaN is rejected rather than coerced, because `NaN -> False` would turn a bar the
    trigger could not evaluate into "did not fire".

    LIMIT OF THIS GUARD, measured rather than assumed. It catches only a mask that still
    CARRIES missingness: a nullable `boolean` Series holding `pd.NA`, or an `object`
    Series holding `None`. It CANNOT catch the common case, because pandas has already
    swallowed it: a plain comparison against NaN evaluates to `False`, not NaN, so

        s = df["close"].rolling(10).mean();  s > s.shift(1)

    returns a clean `bool` dtype whose first ten entries are `False` -- identical, to this
    function and to everything downstream, to a trigger that genuinely did not fire there.
    `test_triggers.py::test_pandas_itself_swallows_warmup_NaN_so_the_guard_cannot_see_it`
    pins that behaviour so this limit stays discoverable.

    CONSEQUENCE, and where it is owed. A trigger with a rolling warm-up contributes its
    warm-up bars to the leaderboard's `base_rate` DENOMINATOR as non-firing bars, which
    understates the base rate. The effect is bounded and small for the required triggers
    (`empty` has no warm-up; `gbb` reads already-populated columns; `geometry`'s DWT
    window is tens of bars against ~160k per pair), but it is a real bias and it is NOT
    fixed here -- `base_rate` is defined in T027, and excluding non-evaluable bars from
    its denominator belongs there, where the denominator lives.
    """
    if not isinstance(out, pd.Series):
        raise TriggerError(
            f"{trigger.name}: evaluate returned {type(out).__name__}, expected pd.Series"
        )
    if len(out) != len(index) or not out.index.equals(index):
        raise TriggerError(
            f"{trigger.name}: evaluate returned a Series misaligned with its input "
            f"({len(out)} values vs {len(index)} bars). Return a mask on the frame's own "
            f"index -- a positionally-correct but differently-indexed mask silently "
            f"misaligns against the outcome table."
        )
    if out.isna().any():
        n = int(out.isna().sum())
        raise TriggerError(
            f"{trigger.name}: evaluate returned {n} NaN value(s). Fill them in the "
            f"trigger itself: coercing NaN to False here would turn an unevaluable "
            f"warm-up bar into 'did not fire' and quietly bias the base rate."
        )
    if out.dtype != bool:
        try:
            out = out.astype(bool)
        except (TypeError, ValueError) as exc:
            raise TriggerError(
                f"{trigger.name}: evaluate returned dtype {out.dtype}, not coercible "
                f"to bool"
            ) from exc
    return out


def main(argv: Iterable[str] | None = None) -> int:
    """`python -m research.outcomes.triggers --list` (contracts/trigger-cli.md)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="list registered triggers")
    ap.add_argument("--no-builtins", action="store_true",
                    help="list only what the caller registered (used by the tests)")
    a = ap.parse_args(list(argv) if argv is not None else None)

    if not a.list:
        ap.print_help()
        return 1
    if not a.no_builtins:
        ensure_builtins()

    registered = all_triggers()
    if not registered:
        print("no triggers registered")
        return 0
    width = max(len(t.name) for t in registered)
    print(f"{'name'.ljust(width)}  {'source':<22}  required columns")
    for t in registered:
        cols = ", ".join(t.required_columns) or "(none)"
        print(f"{t.name.ljust(width)}  {t.source:<22}  {cols}")
        if t.description:
            print(f"{' ' * width}  {'':<22}  {t.description}")
    return 0


if __name__ == "__main__":
    # DELEGATE to the package module rather than run this copy. Under `python -m
    # research.outcomes.triggers` this file is imported TWICE: once as `__main__`, and
    # again as `research.outcomes.triggers` when `builtin_triggers` imports it to reach
    # `register`. Each copy gets its own `_REGISTRY`, so the builtins land in one and
    # `main` reads the other -- which showed up as `--list` reporting "no triggers
    # registered" immediately after registering three. Importing the canonical module
    # here collapses the two into one.
    from research.outcomes.triggers import main as _main

    raise SystemExit(_main())
