"""Block bootstrap for the economic-value interval (T028, research R9).

The required test is that the block version is WIDER than i.i.d. on autocorrelated data.
That is the whole justification for writing a second bootstrap, and without it the claim
"an i.i.d. bootstrap would understate the interval badly" is only an argument. Here it is a
number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "utils"), str(ROOT / "NNNC")):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.outcomes import bootstrap as BS  # noqa: E402

N_RESAMPLES = 2000          # enough for a stable percentile, fast enough for CI


def _ar1(n: int, rho: float, seed: int = 0) -> np.ndarray:
    """An AR(1) series: the simplest thing with the dependence structure the real outcomes
    have, where each bar's value is mostly a repeat of the previous bar's."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho**2) * e[i]
    return x


# =============================================================================================
# THE required test (T028)
# =============================================================================================

@pytest.mark.parametrize("rho", [0.90, 0.98])
def test_block_interval_is_WIDER_than_iid_on_autocorrelated_data(rho):
    """The justification for this module existing. With strongly autocorrelated values, a
    thousand consecutive bars carry far less than a thousand bars' worth of information, and
    i.i.d. resampling does not know that."""
    n = 20_000
    x = _ar1(n, rho, seed=1)
    mask = np.ones(n, dtype=bool)
    block = BS.block_bootstrap_mean([(x, mask)], n_resamples=N_RESAMPLES, seed=7)
    iid = BS.iid_bootstrap_mean([(x, mask)], n_resamples=N_RESAMPLES, seed=7)
    assert block.width > iid.width, (
        f"rho={rho}: block {block.width:.5f} is not wider than iid {iid.width:.5f}"
    )
    assert block.width / iid.width > 1.5, (
        f"rho={rho}: block/iid width ratio {block.width / iid.width:.2f} is implausibly "
        f"small for this much autocorrelation -- check the block length"
    )


def test_the_two_AGREE_on_independent_data():
    """The control. If the block version were simply wider on everything, the test above
    would prove nothing -- it has to be wider only where dependence exists."""
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 20_000)
    mask = np.ones(x.size, dtype=bool)
    block = BS.block_bootstrap_mean([(x, mask)], n_resamples=N_RESAMPLES, seed=7)
    iid = BS.iid_bootstrap_mean([(x, mask)], n_resamples=N_RESAMPLES, seed=7)
    assert 0.75 < block.width / iid.width < 1.35, (
        f"on i.i.d. data the two should roughly agree; ratio {block.width / iid.width:.3f}"
    )


def test_understating_the_interval_is_what_would_manufacture_a_false_positive():
    """The failure mode spelled out as a test. On autocorrelated data with a true mean of
    zero, the i.i.d. interval can exclude zero -- publishable as `selection_value` -- while
    the block interval correctly spans it."""
    base = _ar1(20_000, 0.98, seed=11)
    base = base - base.mean()
    mask = np.ones(base.size, dtype=bool)

    # Self-calibrating: measure both half-widths on the centred series, then shift the mean
    # into the gap BETWEEN them. Hand-picking a shift would silently stop testing anything
    # the moment either interval changed width.
    iid_hw = BS.iid_bootstrap_mean([(base, mask)], n_resamples=N_RESAMPLES, seed=5).width / 2
    blk_hw = BS.block_bootstrap_mean([(base, mask)], n_resamples=N_RESAMPLES, seed=5).width / 2
    assert blk_hw > iid_hw, "no gap to test in; the block interval is not wider here"
    shift = (iid_hw + blk_hw) / 2

    x = base + shift
    block = BS.block_bootstrap_mean([(x, mask)], n_resamples=N_RESAMPLES, seed=5)
    iid = BS.iid_bootstrap_mean([(x, mask)], n_resamples=N_RESAMPLES, seed=5)
    assert not iid.spans_zero, (
        f"fixture too weak: iid must look significant at a shift of {shift:.6f}"
    )
    assert block.spans_zero, "block interval should decline to call this"


# =============================================================================================
# Block length
# =============================================================================================

def test_default_block_length_exceeds_the_measured_mean_hold():
    """T027 measured mean holds of 32-42 bars. A block shorter than one trade leaves the
    dependence straddling block boundaries, which is the thing blocks exist to contain."""
    assert BS.DEFAULT_BLOCK_LENGTH >= 42


def test_a_longer_block_gives_a_wider_interval_on_autocorrelated_data():
    x = _ar1(20_000, 0.98, seed=2)
    mask = np.ones(x.size, dtype=bool)
    widths = [
        BS.block_bootstrap_mean([(x, mask)], block_length=L, n_resamples=N_RESAMPLES,
                                seed=4).width
        for L in (4, 16, 64, 256)
    ]
    assert widths == sorted(widths), f"width should grow with block length, got {widths}"


def test_trailing_partial_block_is_dropped_not_padded():
    """Padding would invent blocks. The discard is at most block_length-1 bars per pair."""
    x = np.ones(100)
    mask = np.ones(100, dtype=bool)
    out = BS.block_bootstrap_mean([(x, mask)], block_length=48, n_resamples=50, seed=0)
    assert out.n_blocks == 2            # 96 bars used, 4 dropped


# =============================================================================================
# Pairing — what FR-014b actually needs
# =============================================================================================

def test_paired_delta_is_NARROWER_than_bootstrapping_the_two_independently():
    """Why the same block draw is applied to both masks. The two triggers share bars, so a
    resample drawing a favourable stretch moves both means together and cancels in the
    difference. Treating them as independent adds their variances and would make a real
    difference look indeterminate."""
    n = 20_000
    x = _ar1(n, 0.95, seed=13)
    rng = np.random.default_rng(17)
    a = rng.random(n) < 0.30
    b = np.ones(n, dtype=bool)

    paired = BS.paired_block_bootstrap_delta([(x, a, b)], n_resamples=N_RESAMPLES, seed=9)
    ia = BS.block_bootstrap_mean([(x, a)], n_resamples=N_RESAMPLES, seed=9)
    ib = BS.block_bootstrap_mean([(x, b)], n_resamples=N_RESAMPLES, seed=9)
    naive_width = float(np.hypot(ia.width, ib.width))
    assert paired.width < naive_width, (
        f"paired {paired.width:.6f} should be narrower than the independent combination "
        f"{naive_width:.6f}"
    )


def test_paired_delta_point_matches_the_difference_of_the_two_means():
    n = 5000
    x = _ar1(n, 0.9, seed=21)
    rng = np.random.default_rng(22)
    a = rng.random(n) < 0.4
    b = np.ones(n, dtype=bool)
    paired = BS.paired_block_bootstrap_delta([(x, a, b)], n_resamples=200, seed=1)
    # computed over the same whole blocks the bootstrap uses, so the trailing partial
    # block is excluded from both sides
    L = BS.DEFAULT_BLOCK_LENGTH
    m = (n // L) * L
    expected = x[:m][a[:m]].mean() - x[:m][b[:m]].mean()
    assert paired.point == pytest.approx(expected, abs=1e-12)


def test_identical_masks_give_a_delta_of_exactly_zero_with_a_degenerate_interval():
    """A sanity floor: a trigger compared with itself must not show a difference, and the
    interval must collapse rather than wander."""
    x = _ar1(5000, 0.9, seed=31)
    m = np.ones(x.size, dtype=bool)
    out = BS.paired_block_bootstrap_delta([(x, m, m)], n_resamples=200, seed=2)
    assert out.point == pytest.approx(0.0, abs=1e-15)
    assert out.width == pytest.approx(0.0, abs=1e-12)
    assert out.spans_zero


# =============================================================================================
# Groups must not be resampled across their seam
# =============================================================================================

def test_blocks_never_span_a_group_boundary():
    """Two pairs' bars are adjacent in a concatenated array but NOT adjacent in time.
    Resampling across the seam would manufacture dependence that does not exist.

    Constructed so a seam-spanning block would be detectable: each group is exactly two
    blocks long, so correct handling gives 4 blocks and concatenating first would give 4
    as well -- but the VALUES differ, so a seam block would mix +1 and -1 and shrink the
    interval. Here the two groups are constants, so the correct interval must contain only
    the three attainable means (-1, 0, +1 weighted), never a value from a mixed block.
    """
    L = 8
    g1 = (np.ones(2 * L), np.ones(2 * L, dtype=bool))
    g2 = (-np.ones(2 * L), np.ones(2 * L, dtype=bool))
    out = BS.block_bootstrap_mean([g1, g2], block_length=L, n_resamples=2000, seed=0)
    assert out.n_blocks == 4
    assert out.point == pytest.approx(0.0)
    # every block is wholly +1 or wholly -1, so a resample mean is always k/4 - (4-k)/4
    assert out.lo == pytest.approx(-1.0) and out.hi == pytest.approx(1.0)


# =============================================================================================
# Degenerate inputs
# =============================================================================================

def test_an_empty_mask_returns_nan_rather_than_crashing():
    x = np.ones(500)
    out = BS.block_bootstrap_mean([(x, np.zeros(500, dtype=bool))], n_resamples=50)
    assert np.isnan(out.point) and np.isnan(out.lo)


def test_a_series_shorter_than_one_block_yields_no_blocks():
    out = BS.block_bootstrap_mean([(np.ones(10), np.ones(10, dtype=bool))],
                                  block_length=48, n_resamples=50)
    assert out.n_blocks == 0 and np.isnan(out.point)


def test_a_sparse_mask_records_degenerate_resamples_rather_than_hiding_them():
    """With very few selected bars some resamples draw none. Those are excluded from the
    percentile and COUNTED, so a reader can see the interval rests on fewer draws than
    requested instead of silently trusting it."""
    n = 48 * 40
    x = np.ones(n)
    mask = np.zeros(n, dtype=bool)
    mask[0] = True                      # exactly one selected bar, in block 0
    out = BS.block_bootstrap_mean([(x, mask)], n_resamples=500, seed=0)
    assert out.n_degenerate > 0
    assert out.point == pytest.approx(1.0)


def test_results_are_reproducible_for_a_given_seed():
    x = _ar1(5000, 0.9, seed=41)
    m = np.ones(x.size, dtype=bool)
    a = BS.block_bootstrap_mean([(x, m)], n_resamples=300, seed=99)
    b = BS.block_bootstrap_mean([(x, m)], n_resamples=300, seed=99)
    assert (a.lo, a.hi) == (b.lo, b.hi)
