"""Economic value from the table alone, with NO model (T027, FR-014, FR-009b, SC-010).

This is the first of the leaderboard's two columns. It answers: if you traded EVERY bar a
trigger fires on, under the one shared exit policy, what did you realise? No model, no
threshold, no training — just the trigger and the table.

That makes it the column that can distinguish the feature's two rival explanations. A
trigger that selects moments when a forward move is collectable shows value here with no
model at all. A trigger that merely hands a model an easy target shows nothing here, and
shows up only in the learnability column (T029).

WHY THIS IS A BOUND AND NOT A RETURN, stated once and meant throughout (FR-017). Trading
every triggered bar is not realisable: positions would overlap, capital is finite, and the
table simulates each bar independently as though it were the only trade. So this number is
an upper bound on what the trigger could be worth, useful for RANKING triggers against each
other and against the `empty` null. Any claim about actual P&L comes from `evaluation/` and
its ~3-4pp/yr floor, never from here.

THE NUMBER THAT MOTIVATES THE WHOLE COLUMN. Over SOL/USDT's 164,918 bars the table's mean
realised outcome is **-0.001427** while its median is **+0.005595**, and 56.19% of bars have
a positive outcome. Hit rate and economic value therefore point in OPPOSITE directions: most
bars win a little, a minority lose a lot. Any column built on the hit rate alone -- which is
what label precision is -- would call this profitable. It is not. That asymmetry is the
reason FR-014 asks for the realised outcome and its distribution rather than a success count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from research.outcomes import bootstrap as _bootstrap
from research.outcomes import labels as _labels
from research.outcomes import triggers as _triggers
from research.outcomes.leakage import (
    OUTCOME_COLUMN_CLOSE_REASON,
    OUTCOME_COLUMN_HOLD_BARS,
    OUTCOME_COLUMN_FILLABLE,
    OUTCOME_COLUMN_GAPFILL_SHARE,
    OUTCOME_COLUMN_PROFIT,
)

#: Rows whose forward window ran off the end of the series. `simulate_forward` returns
#: `(0.0, ..., CLOSE_END_OF_DATA)` for these -- an ARTIFICIAL zero, not a realised outcome
#: of 0.0. Including them would pull every mean toward zero by their share, so they are
#: excluded from every statistic and counted separately. Only 42 of 164,918 rows on
#: SOL/USDT (0.0255%), but the exclusion is a correctness point, not a materiality one: a
#: fabricated zero is not a measurement, however few there are.
EXCLUDED_CLOSE_REASON = "end_of_data"

#: Reported quantiles. The distribution is required by FR-014 rather than optional, because
#: the mean alone cannot distinguish "many small wins and a few catastrophes" from "a steady
#: small edge", and those two call for opposite decisions.
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True)
class OutcomeStats:
    """What the table says about one trigger's bars. No model involved."""

    n_triggered: int
    n_scored: int                 # after dropping end_of_data
    n_excluded_end_of_data: int
    base_rate: float              # share of ALL bars the trigger fires on
    mean: float                   # EV over every triggered bar (FR-014, as specified)
    median: float
    std: float
    win_rate: float               # share of scored bars with a positive outcome
    #: EV over NON-OVERLAPPING triggered bars -- the comparable figure. See
    #: `CLUSTERING_FINDING`.
    n_non_overlapping: int = 0
    mean_non_overlapping: float = float("nan")
    win_rate_non_overlapping: float = float("nan")
    quantiles: dict[float, float] = field(default_factory=dict)
    gapfill_share: float = 0.0    # FR-020, SC-009

    @property
    def is_empty(self) -> bool:
        return self.n_scored == 0

    @property
    def cluster_ratio(self) -> float:
        """Distinct opportunities per triggered bar. 1.0 = no clustering."""
        return self.n_non_overlapping / self.n_scored if self.n_scored else float("nan")


def _validate(features: pd.DataFrame, outcomes: pd.DataFrame) -> None:
    _labels.assert_aligned(features, outcomes)
    missing = [
        c for c in (OUTCOME_COLUMN_PROFIT, OUTCOME_COLUMN_CLOSE_REASON)
        if c not in outcomes.columns
    ]
    if missing:
        raise KeyError(
            f"outcomes frame is missing {missing}; build the table first "
            f"(research.outcomes.table)"
        )


def _select(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    trigger: "_triggers.Trigger | str",
    *,
    fillable_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """One pair's selected outcomes.

    Returns `(profit_all, profit_non_overlapping, gapfill_all, n_triggered, n_bars)`.

    Kept per-pair deliberately: the non-overlapping scan walks bars in time order, so it is
    only meaningful WITHIN one pair's contiguous series. Concatenating pairs first and
    scanning the result would let one pair's last trade suppress the next pair's first.
    """
    _validate(features, outcomes)
    fired = _triggers.evaluate(trigger, features).to_numpy()
    n_bars = int(fired.size)
    # Counted BEFORE the fillable filter: `base_rate` describes the TRIGGER, not the
    # fillable subset. Conflating them would make a trigger look rarer than it is on an
    # illiquid pair, and the base rate is how a reader judges whether a trigger is
    # selective at all.
    n_fired_all = int(fired.sum())
    if fillable_only:
        if OUTCOME_COLUMN_FILLABLE not in outcomes.columns:
            raise KeyError(
                f"fillable_only requires {OUTCOME_COLUMN_FILLABLE!r} in the outcomes frame"
            )
        fired = fired & outcomes[OUTCOME_COLUMN_FILLABLE].to_numpy(dtype=bool)
    n_triggered = int(fired.sum())

    usable = fired & (
        outcomes[OUTCOME_COLUMN_CLOSE_REASON].to_numpy() != EXCLUDED_CLOSE_REASON
    )
    idx = np.flatnonzero(usable)
    profit = outcomes[OUTCOME_COLUMN_PROFIT].to_numpy(dtype=float)
    hold = outcomes[OUTCOME_COLUMN_HOLD_BARS].to_numpy(dtype=float)
    gap = (
        outcomes[OUTCOME_COLUMN_GAPFILL_SHARE].to_numpy(dtype=float)
        if OUTCOME_COLUMN_GAPFILL_SHARE in outcomes.columns
        else np.zeros_like(profit)
    )

    # Greedy earliest-first non-overlapping selection: take a trade, then skip every bar
    # until it closes. Greedy is the right choice here precisely BECAUSE it is not
    # optimising -- picking the best subset would be a lookahead selection of its own, and
    # the point is to count distinct OPPORTUNITIES, not to construct a favourable schedule.
    keep: list[int] = []
    free = -1
    for i in idx:
        if i > free:
            keep.append(int(i))
            free = int(i) + max(int(hold[i]), 0)

    return (profit[idx],
            profit[np.asarray(keep, dtype=int)] if keep else np.empty(0),
            gap[idx], n_triggered, n_bars, n_fired_all)


def _stats(
    profit: np.ndarray,
    profit_nv: np.ndarray,
    gap: np.ndarray,
    n_triggered: int,
    n_bars: int,
    n_fired_all: int,
) -> OutcomeStats:
    base_rate = n_fired_all / n_bars if n_bars else 0.0
    n_scored = int(profit.size)
    if n_scored == 0:
        return OutcomeStats(
            n_triggered=n_triggered, n_scored=0, n_excluded_end_of_data=n_triggered,
            base_rate=base_rate, mean=float("nan"), median=float("nan"),
            std=float("nan"), win_rate=float("nan"),
            quantiles={q: float("nan") for q in QUANTILES},
        )
    return OutcomeStats(
        n_triggered=n_triggered,
        n_scored=n_scored,
        n_excluded_end_of_data=n_triggered - n_scored,
        base_rate=base_rate,
        mean=float(profit.mean()),
        median=float(np.median(profit)),
        std=float(profit.std(ddof=1)) if n_scored > 1 else 0.0,
        win_rate=float((profit > 0.0).mean()),
        n_non_overlapping=int(profit_nv.size),
        mean_non_overlapping=float(profit_nv.mean()) if profit_nv.size else float("nan"),
        win_rate_non_overlapping=(
            float((profit_nv > 0.0).mean()) if profit_nv.size else float("nan")
        ),
        quantiles={q: float(np.quantile(profit, q)) for q in QUANTILES},
        gapfill_share=float(gap.mean()) if gap.size else 0.0,
    )


def economic_value(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    trigger: "_triggers.Trigger | str",
    *,
    fillable_only: bool = False,
) -> OutcomeStats:
    """Mean realised outcome over the bars `trigger` fires on, for ONE pair (FR-014).

    `fillable_only` restricts to bars the table marked fillable (FR-009b). That matters far
    more than it sounds: only **28.17%** of SOL/USDT bars are fillable, so this is not a
    rounding adjustment but a different population. Reporting the unfiltered figure alone
    would let a trigger earn its value on bars no one could have traded -- which this
    repository has been bitten by once, when trades whose stake exceeded available volume
    inflated an out-of-sample result until the volume filter was turned on. FR-009b
    therefore requires BOTH, always, and the leaderboard prints them side by side.

    `base_rate` is over ALL bars, including a trigger's rolling warm-up, which contributes
    to the denominator as non-firing. Quantified rather than engineered around:
    `breakout`'s warm-up is 20 bars and `geometry`'s is 9, against ~165k bars per pair -- a
    bias of about 0.012%, three orders of magnitude below anything the verdict rule can
    resolve. (T022 flagged this as needing a fix here; measuring it WAS the fix.)
    """
    return _stats(*_select(features, outcomes, trigger, fillable_only=fillable_only))


def pooled_economic_value(
    frames: Iterable[tuple[pd.DataFrame, pd.DataFrame]],
    trigger: "_triggers.Trigger | str",
    *,
    fillable_only: bool = False,
) -> OutcomeStats:
    """`economic_value` over several pairs, pooled on the BAR.

    Pooling bars rather than averaging per-pair means is deliberate: a per-pair average
    weights a 101k-bar pair equally with a 165k-bar one, and this repository's edge is
    already known to concentrate in particular high-volatility names. A bar-pooled mean
    lets that concentration show in the number instead of being hidden by equal weighting
    -- which is what a leaderboard reader needs to see.

    The non-overlapping selection is made PER PAIR and then pooled, never across the
    concatenation, so one pair's final trade cannot suppress the next pair's first.
    """
    parts = [
        _select(f, o, trigger, fillable_only=fillable_only) for f, o in frames
    ]
    if not parts:
        raise ValueError("no frames supplied")
    return _stats(
        np.concatenate([p[0] for p in parts]) if parts else np.empty(0),
        np.concatenate([p[1] for p in parts]) if parts else np.empty(0),
        np.concatenate([p[2] for p in parts]) if parts else np.empty(0),
        sum(p[3] for p in parts),
        sum(p[4] for p in parts),
        sum(p[5] for p in parts),
    )


def format_row(name: str, unfiltered: OutcomeStats, fillable: OutcomeStats) -> str:
    """One leaderboard line. Both populations side by side, never combined (FR-016)."""
    if unfiltered.is_empty:
        return f"{name:<14}{'(no scored bars)':>60}"
    return (
        f"{name:<14}{unfiltered.base_rate * 100:>9.3f}%{unfiltered.n_scored:>10}"
        f"{unfiltered.n_non_overlapping:>9}{unfiltered.cluster_ratio:>7.3f}"
        f"{unfiltered.mean:>+12.6f}{unfiltered.mean_non_overlapping:>+12.6f}"
        f"{fillable.mean_non_overlapping:>+13.6f}"
        f"{unfiltered.win_rate * 100:>9.2f}%{unfiltered.gapfill_share * 100:>9.3f}%"
    )


def header() -> str:
    return (
        f"{'trigger':<14}{'base rate':>10}{'n scored':>10}{'n n-ovl':>9}{'ratio':>7}"
        f"{'EV all':>12}{'EV n-ovl':>12}{'EV fill n-ovl':>13}{'win rate':>10}"
        f"{'gapfill':>10}"
    )


# =================================================================================================
# Intervals (T028). The gate on interpreting any ranking above.
# =================================================================================================

def _delta_groups(
    frames: Iterable[tuple[pd.DataFrame, pd.DataFrame]],
    trigger: "_triggers.Trigger | str",
    reference: "_triggers.Trigger | str",
    *,
    fillable_only: bool,
    non_overlapping: bool,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-pair `(profit, mask_trigger, mask_reference)` for the paired block bootstrap.

    Both masks are built over the SAME bars of the same pair, which is what makes the
    bootstrap paired and lets the shared market component cancel out of the difference.
    """
    groups = []
    for f, o in frames:
        _validate(f, o)
        profit = o[OUTCOME_COLUMN_PROFIT].to_numpy(dtype=float)
        usable = o[OUTCOME_COLUMN_CLOSE_REASON].to_numpy() != EXCLUDED_CLOSE_REASON
        if fillable_only:
            usable = usable & o[OUTCOME_COLUMN_FILLABLE].to_numpy(dtype=bool)
        hold = o[OUTCOME_COLUMN_HOLD_BARS].to_numpy(dtype=float)
        masks = []
        for t in (trigger, reference):
            m = _triggers.evaluate(t, f).to_numpy() & usable
            if non_overlapping:
                keep = np.zeros_like(m)
                free = -1
                for i in np.flatnonzero(m):
                    if i > free:
                        keep[i] = True
                        free = int(i) + max(int(hold[i]), 0)
                m = keep
            masks.append(m)
        groups.append((profit, masks[0], masks[1]))
    return groups


def economic_value_delta(
    frames: Iterable[tuple[pd.DataFrame, pd.DataFrame]],
    trigger: "_triggers.Trigger | str",
    reference: "_triggers.Trigger | str" = "empty",
    *,
    fillable_only: bool = False,
    non_overlapping: bool = True,
    block_length: int = _bootstrap.DEFAULT_BLOCK_LENGTH,
    n_resamples: int = _bootstrap.DEFAULT_N_RESAMPLES,
    seed: int = 0,
) -> "_bootstrap.Interval":
    """The quantity FR-014b's verdict rule turns on: `EV(trigger) - EV(reference)`, with a
    block-bootstrap interval.

    `non_overlapping=True` by default, because T027 established that the all-bars mean is
    not comparable across triggers -- it is weighted by how often a trigger repeats itself,
    and correcting for that flipped `gbb` from positive to negative. The verdict rule must
    run on the comparable figure, so that is the default rather than an option.
    """
    return _bootstrap.paired_block_bootstrap_delta(
        _delta_groups(frames, trigger, reference,
                      fillable_only=fillable_only, non_overlapping=non_overlapping),
        block_length=block_length, n_resamples=n_resamples, seed=seed,
    )


# =================================================================================================
# The verdict rule (T030, FR-014b as amended by FR-015b, SC-007)
# =================================================================================================

VERDICT_SELECTION_VALUE = "selection_value"
VERDICT_DECORATION = "decoration"
VERDICT_INDETERMINATE = "indeterminate"

#: The null. Not a parameter: FR-014a makes `empty` the reference precisely so no threshold
#: has to be agreed in advance, and allowing a different reference would reintroduce one.
REFERENCE_TRIGGER = "empty"


@dataclass(frozen=True)
class Verdict:
    """One trigger's adjudication, with everything the rule consumed (FR-014b)."""

    trigger: str
    verdict: str
    ev_delta: float
    ev_lo: float
    ev_hi: float
    ev_beats_null: bool
    learn_delta: float
    learn_se: float
    learn_beats_null: bool

    def __str__(self) -> str:
        return (
            f"{self.trigger:<14}{self.verdict:<18}"
            f"EV {self.ev_delta:+.6f} [{self.ev_lo:+.6f},{self.ev_hi:+.6f}]"
            f"{'  BEATS' if self.ev_beats_null else '      --'}   "
            f"cond-MCC {self.learn_delta:+.4f} +/-{self.learn_se:.4f}"
            f"{'  BEATS' if self.learn_beats_null else '      --'}"
        )


def assign_verdict(
    trigger: str,
    ev_delta: "_bootstrap.Interval",
    learn_delta: float,
    learn_se: float,
) -> Verdict:
    """FR-014b, applied. TOTAL by construction: three branches, the third unconditional.

    `ev_delta` is the paired block-bootstrap interval on `EV(trigger) - EV(empty)`;
    "exceeds by more than the measurement uncertainty" is read as its 95% interval lying
    wholly above zero. `learn_delta`/`learn_se` are the CONDITIONAL learnability difference
    and its standard error (FR-015b) -- never the unconditional figure, which T029 showed
    measures trigger recovery and would make this branch fire on every trigger.

    A NOTE ON AN ASYMMETRY IN THE SPEC, implemented as written rather than silently
    "corrected": the economic-value bar is a 95% interval while the learnability bar is ONE
    standard error (contracts/trigger-cli.md: "by more than the SE on that difference"). One
    SE is roughly a 68% bar, so `decoration` is EASIER to earn than `selection_value`. That
    is a deliberate reading of the spec's own words; it is reported alongside each verdict so
    a reader can see which bar a row cleared rather than inferring it.
    """
    ev_beats = bool(ev_delta.lo > 0.0)
    learn_beats = bool(learn_delta > learn_se)

    if ev_beats:
        verdict = VERDICT_SELECTION_VALUE
    elif learn_beats:
        verdict = VERDICT_DECORATION
    else:
        verdict = VERDICT_INDETERMINATE

    return Verdict(
        trigger=trigger, verdict=verdict,
        ev_delta=ev_delta.point, ev_lo=ev_delta.lo, ev_hi=ev_delta.hi,
        ev_beats_null=ev_beats,
        learn_delta=learn_delta, learn_se=learn_se, learn_beats_null=learn_beats,
    )


def verdict_header() -> str:
    return (
        f"{'trigger':<14}{'verdict':<18}{'economic value vs null (95% CI)':<48}"
        f"{'conditional MCC vs null (+/-1 SE)'}"
    )
