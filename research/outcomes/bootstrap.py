"""Block bootstrap for the economic-value interval (T028, research R9).

FR-014b needs the uncertainty on the DIFFERENCE between a trigger's economic value and the
`empty` null's, because its verdict turns on whether that difference exceeds its own
measurement error. This module supplies it.

WHY BLOCKS, now with the numbers rather than the argument. Adjacent bars' outcomes are not
independent: each bar's outcome is the result of simulating forward from it, and those
forward windows overlap almost completely. T027 measured the overlap directly -- mean hold
is **32 to 42 bars** depending on the trigger, and the ratio of distinct non-overlapping
opportunities to triggered bars runs from **0.027** (`empty`) to **0.509** (`geometry`). So
a thousand consecutive `empty` bars are not a thousand observations; they are closer to
twenty-seven. An i.i.d. bootstrap would treat them as a thousand and return an interval
roughly sqrt(1/0.027) ~ 6x too narrow, which is exactly how an indeterminate result becomes
a false `selection_value`.

WHY NOT `evaluation/bootstrap.py`. That module resamples paired TRADE RECORDS i.i.d. for the
pooled harness, which is sound there: harness trades are already non-overlapping by
construction (one position per pair at a time). Here the unit is the BAR and the bars
overlap, so it is a different statistic and reusing the function would import an
independence assumption that does not hold. Deliberately a second implementation, not a
duplicated one -- the two compute different things.

NON-OVERLAPPING blocks, not moving blocks. Both are standard; non-overlapping is chosen
because it makes the per-block sums precomputable, which is what makes this affordable: a
resample becomes a gather over ~35,000 block aggregates instead of a gather over 1.7 million
bars, so 2,000 resamples run in seconds rather than hours. The cost is a slightly coarser
treatment of the series ends, which is immaterial at 1.7M bars.

THESE INTERVALS ARE NOT THE HARNESS'S +/-3-4pp/yr (R9's explicit non-goal, FR-017). That
floor governs P&L claims. The table's economic value is a different quantity measured a
different way, and treating a lookahead bound as a realisable return is the error this
feature exists to avoid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

#: Default block length in bars. Chosen to exceed the measured mean hold (32-42 bars, T027)
#: so that a block spans at least one full trade and the dependence it is meant to capture
#: falls INSIDE a block rather than across block boundaries. 48 bars = 12 hours at 15m.
#: A block shorter than the mean hold would leave the overlap straddling blocks, which is
#: the failure the block bootstrap exists to avoid.
DEFAULT_BLOCK_LENGTH = 48

#: Matches `evaluation/bootstrap.py`'s count, so the two intervals are not differently
#: resolved for an incidental reason.
DEFAULT_N_RESAMPLES = 20_000

DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile interval and the diagnostics to judge it."""

    point: float
    lo: float
    hi: float
    alpha: float
    n_blocks: int
    block_length: int
    n_resamples: int
    n_degenerate: int = 0          # resamples that drew no usable bar

    @property
    def spans_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def __str__(self) -> str:
        return f"{self.point:+.6f} [{self.lo:+.6f}, {self.hi:+.6f}]"


def _block_aggregates(
    values: np.ndarray, mask: np.ndarray, block_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-block (sum, count) of `values` over the bars `mask` selects.

    Trailing bars that do not fill a whole block are dropped. Documented rather than
    padded: padding would invent blocks, and at 1.7M bars the discard is at most 47 bars
    per pair.
    """
    n = (len(values) // block_length) * block_length
    if n == 0:
        return np.zeros(0), np.zeros(0)
    sel = mask[:n].astype(bool)
    v = np.where(sel, values[:n], 0.0).reshape(-1, block_length)
    c = sel.reshape(-1, block_length)
    return v.sum(axis=1), c.sum(axis=1).astype(float)


def _percentile_interval(
    draws: np.ndarray, point: float, alpha: float, *, n_blocks: int,
    block_length: int, n_resamples: int, n_degenerate: int,
) -> Interval:
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(
        point=point, lo=float(lo), hi=float(hi), alpha=alpha, n_blocks=n_blocks,
        block_length=block_length, n_resamples=n_resamples, n_degenerate=n_degenerate,
    )


def block_bootstrap_mean(
    groups: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Interval:
    """Interval on the mean of `values[mask]`, pooled across `groups`.

    `groups` is one `(values, mask)` per pair. Blocks are formed WITHIN a group and pooled
    afterwards, so a block never spans a pair boundary -- two pairs' bars are adjacent in a
    concatenated array but not adjacent in time, and resampling across the seam would
    manufacture dependence that does not exist.
    """
    sums, counts = [], []
    for values, mask in groups:
        s, c = _block_aggregates(np.asarray(values, float), np.asarray(mask), block_length)
        sums.append(s)
        counts.append(c)
    bs = np.concatenate(sums) if sums else np.zeros(0)
    bc = np.concatenate(counts) if counts else np.zeros(0)
    n_blocks = int(bs.size)
    if n_blocks == 0 or bc.sum() == 0:
        return Interval(float("nan"), float("nan"), float("nan"), alpha, n_blocks,
                        block_length, n_resamples)

    point = float(bs.sum() / bc.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    tot_c = bc[idx].sum(axis=1)
    tot_s = bs[idx].sum(axis=1)
    ok = tot_c > 0
    draws = np.divide(tot_s, tot_c, out=np.full(n_resamples, np.nan), where=ok)
    return _percentile_interval(
        draws[ok], point, alpha, n_blocks=n_blocks, block_length=block_length,
        n_resamples=n_resamples, n_degenerate=int((~ok).sum()),
    )


def paired_block_bootstrap_delta(
    groups: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Interval:
    """Interval on `mean(values[mask_a]) - mean(values[mask_b])`, the quantity FR-014b needs.

    `groups` is one `(values, mask_a, mask_b)` per pair. The SAME block draw is applied to
    both masks, which is what makes this paired: the two triggers are measured over the same
    bars, so a resample that happens to draw a favourable stretch moves both means together
    and the difference is unaffected. Bootstrapping the two independently would add their
    variances instead of cancelling the shared component, inflating the interval and making
    a real difference look indeterminate.
    """
    sa, ca, sb, cb = [], [], [], []
    for values, mask_a, mask_b in groups:
        v = np.asarray(values, float)
        s1, c1 = _block_aggregates(v, np.asarray(mask_a), block_length)
        s2, c2 = _block_aggregates(v, np.asarray(mask_b), block_length)
        sa.append(s1); ca.append(c1); sb.append(s2); cb.append(c2)
    BSA = np.concatenate(sa) if sa else np.zeros(0)
    BCA = np.concatenate(ca) if ca else np.zeros(0)
    BSB = np.concatenate(sb) if sb else np.zeros(0)
    BCB = np.concatenate(cb) if cb else np.zeros(0)
    n_blocks = int(BSA.size)
    if n_blocks == 0 or BCA.sum() == 0 or BCB.sum() == 0:
        return Interval(float("nan"), float("nan"), float("nan"), alpha, n_blocks,
                        block_length, n_resamples)

    point = float(BSA.sum() / BCA.sum() - BSB.sum() / BCB.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    ca_t, cb_t = BCA[idx].sum(axis=1), BCB[idx].sum(axis=1)
    ok = (ca_t > 0) & (cb_t > 0)
    draws = np.full(n_resamples, np.nan)
    np.divide(BSA[idx].sum(axis=1), ca_t, out=draws, where=ok)
    other = np.full(n_resamples, np.nan)
    np.divide(BSB[idx].sum(axis=1), cb_t, out=other, where=ok)
    delta = draws - other
    return _percentile_interval(
        delta[ok], point, alpha, n_blocks=n_blocks, block_length=block_length,
        n_resamples=n_resamples, n_degenerate=int((~ok).sum()),
    )


def iid_bootstrap_mean(
    groups: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Interval:
    """The WRONG interval, kept deliberately.

    Resamples selected bars independently, ignoring the overlap. Exists so the block
    version's necessity is a measured comparison rather than a claim in a docstring, and so
    the ratio between the two can be reported: an under-stated interval is the specific
    mechanism by which a null result would be published as `selection_value`. Never use it
    for a verdict.
    """
    vals = [np.asarray(v, float)[np.asarray(m, bool)] for v, m in groups]
    x = np.concatenate(vals) if vals else np.zeros(0)
    if x.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), alpha, 0, 1, n_resamples)
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, x.size, size=(n_resamples, x.size))].mean(axis=1)
    return _percentile_interval(
        draws, float(x.mean()), alpha, n_blocks=int(x.size), block_length=1,
        n_resamples=n_resamples, n_degenerate=0,
    )


# =================================================================================================
# CALENDAR blocks (2026-10-09, T035 items 1 and 2)
# =================================================================================================

#: A block length that contains the FULL hold, not just the typical one. `max_hold_days` is
#: 10 days = 960 bars at 15m, so a 48-bar block leaves a trade's dependence straddling 20
#: blocks. `DEFAULT_BLOCK_LENGTH` (48) was chosen against the MEAN hold (32-42 bars), which
#: is the wrong statistic: the block has to contain the longest dependence it is meant to
#: absorb, not the average one.
#:
#: Sensitivity across L is reported rather than one value asserted, because the right answer
#: is a judgement and the honest output is how much the verdict moves.
FULL_HOLD_BLOCK_LENGTH = 960


def _calendar_block_ids(dates: np.ndarray, block_length: int, bar_ns: int) -> np.ndarray:
    """Map timestamps to a GLOBAL block id shared by every pair.

    This is the fix for resampling pairs independently. Same-time crypto bars are highly
    correlated -- an 11-pair cross-section during one market event is close to ONE
    observation -- yet per-pair blocks let a resample draw pair A's May-2022 block alongside
    pair B's March-2024 block, implicitly treating the cross-section as independent. A
    calendar block draws a TIME WINDOW and takes every pair's bars inside it together, so
    the cross-sectional correlation is carried rather than averaged away.
    """
    return (dates.astype("datetime64[ns]").astype(np.int64) // (bar_ns * block_length))


def paired_calendar_block_delta(
    groups: "Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]",
    *,
    block_length: int = FULL_HOLD_BLOCK_LENGTH,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    bar_minutes: int = 15,
) -> Interval:
    """`paired_block_bootstrap_delta` with blocks keyed on CALENDAR TIME across all pairs.

    `groups` is one `(values, mask_a, mask_b, dates)` per pair. Blocks are global time
    windows, so one draw moves every pair's bars in that window together.

    Expect WIDER intervals than the per-pair version: it stops crediting the cross-section
    with independence it does not have. If a verdict survives this and the per-pair version
    disagrees, the per-pair one was overconfident.
    """
    bar_ns = int(bar_minutes) * 60 * 1_000_000_000
    acc: dict[int, list[float]] = {}
    for values, mask_a, mask_b, dates in groups:
        v = np.asarray(values, float)
        d = np.asarray(dates)
        ids = _calendar_block_ids(d, block_length, bar_ns)
        for mask, slot in ((np.asarray(mask_a, bool), 0), (np.asarray(mask_b, bool), 2)):
            if not mask.any():
                continue
            bid = ids[mask]
            vals = v[mask]
            for b, x in zip(bid, vals):
                row = acc.setdefault(int(b), [0.0, 0.0, 0.0, 0.0])
                row[slot] += float(x)
                row[slot + 1] += 1.0

    if not acc:
        return Interval(float("nan"), float("nan"), float("nan"), alpha, 0,
                        block_length, n_resamples)

    arr = np.array([acc[k] for k in sorted(acc)], dtype=float)
    BSA, BCA, BSB, BCB = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    n_blocks = int(arr.shape[0])
    if BCA.sum() == 0 or BCB.sum() == 0:
        return Interval(float("nan"), float("nan"), float("nan"), alpha, n_blocks,
                        block_length, n_resamples)

    point = float(BSA.sum() / BCA.sum() - BSB.sum() / BCB.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    ca, cb = BCA[idx].sum(axis=1), BCB[idx].sum(axis=1)
    ok = (ca > 0) & (cb > 0)
    a = np.full(n_resamples, np.nan); b = np.full(n_resamples, np.nan)
    np.divide(BSA[idx].sum(axis=1), ca, out=a, where=ok)
    np.divide(BSB[idx].sum(axis=1), cb, out=b, where=ok)
    delta = a - b
    return _percentile_interval(
        delta[ok], point, alpha, n_blocks=n_blocks, block_length=block_length,
        n_resamples=n_resamples, n_degenerate=int((~ok).sum()),
    )
