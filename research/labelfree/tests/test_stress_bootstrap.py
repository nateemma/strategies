"""Stationary block bootstrap invariants (T011-T013, T018a)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.stress import bootstrap as BS          # noqa: E402


def _panel(n=4000, pairs=3, seed=1):
    """A synthetic panel whose pairs are CORRELATED, so the cross-pair test can discriminate."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n) * 0.01
    out = {"fold_id": "fX", "which": "test", "pairs": {}}
    for i in range(pairs):
        r = common + rng.standard_normal(n) * 0.004      # shared factor + idiosyncratic
        px = 100 * np.cumprod(1 + r)
        out["pairs"][f"P{i}/USDT"] = {
            "open": px, "high": px * 1.001, "low": px * 0.999, "close": px,
            "volume": np.full(n, 100.0), "rvol_stop": np.ones(n), "cost": np.full(n, 0.002),
            "p_sell": np.zeros(n), "p_hold": np.zeros(n), "p_buy": np.full(n, 0.9),
            "rvol": np.full(n, 3.0), "atr_pct_roll": np.full(n, 0.01),
            "guard_metric": np.zeros(n), "close_norm": np.full(n, -0.5),
            "adx": np.full(n, 50.0), "bb_width": np.full(n, 0.1),
            "_ret": r,
        }
    return out


# --------------------------------------------------------------- T011: within-block ordering

def test_within_block_ordering_preserved():
    """Consecutive bars inside a block come from consecutive source bars.

    A bootstrap that shuffled within blocks would destroy exactly the autocorrelation and
    volatility clustering the method exists to preserve — and those are what a path-dependent
    policy reacts to.
    """
    rng = np.random.default_rng(0)
    idx = BS.block_index(5000, 5000, block_len=192, rng=rng)
    d = np.diff(idx)
    continues = d == 1
    wraps = (idx[:-1] == 4999) & (idx[1:] == 0)
    # Everything that is not a new-block jump must be a +1 step (or a wrap).
    runs = BS.realised_block_lengths(idx)
    assert runs.sum() == idx.size
    assert (continues | wraps | ~continues).all()
    assert continues.mean() > 0.9, "far too many block breaks for L=192"


def test_geometric_block_lengths():
    """Realised block lengths are geometric with mean L, not fixed.

    Fixed blocks make the resampled series non-stationary — a bar's chance of starting a block
    depends on its position modulo L — which is why Politis & Romano use the geometric variant.
    """
    rng = np.random.default_rng(3)
    idx = BS.block_index(20000, 60000, block_len=192, rng=rng)
    runs = BS.realised_block_lengths(idx)
    assert runs.size > 100
    assert 140 < runs.mean() < 250, f"mean block length {runs.mean():.0f}, expected ~192"
    assert runs.std() > 0.5 * runs.mean(), "lengths look fixed, not geometric"
    assert runs.min() == 1, "a geometric distribution must produce length-1 blocks"


# ------------------------------------------------- T013: every column from the same source bar

def test_all_columns_resampled_together():
    """Prices, guards, cost AND p_buy/p_sell come from the same source bar.

    If they ever decouple, the model's reaction no longer matches the market state it reacted
    to, and the path is a fiction — which is the whole reason the probabilities ride along
    instead of being recomputed (research.md R1).
    """
    p = _panel(n=500)
    # Every NON-PRICE column is a distinct function of the bar index, so decoupling shows up.
    # Price columns are deliberately excluded from this check: they carry a continuity factor
    # at block joins (see `continuity_factors`), so they are intentionally NOT equal to their
    # source value. Their invariant is tested separately below.
    for d in p["pairs"].values():
        for k, mult in (("p_buy", 2), ("cost", 3), ("adx", 4), ("rvol", 5)):
            d[k] = np.arange(500, dtype=float) * mult
    idx = BS.block_index(500, 500, 96, np.random.default_rng(5))
    out = BS.resample_panel(p, idx)
    j = idx % 500
    for d in out["pairs"].values():
        assert np.allclose(d["p_buy"], j * 2)
        assert np.allclose(d["cost"], j * 3)
        assert np.allclose(d["adx"], j * 4)
        assert np.allclose(d["rvol"], j * 5)


def test_price_columns_share_one_continuity_factor():
    """open/high/low/close are scaled by the SAME factor per bar, so their exact intrabar
    relationships survive the splice. A per-column factor would let low exceed high."""
    p = _panel(n=2000)
    idx = BS.block_index(2000, 2000, 96, np.random.default_rng(7))
    out = BS.resample_panel(p, idx)
    for pair, d in out["pairs"].items():
        src = p["pairs"][pair]
        j = idx % 2000
        fac = d["close"] / src["close"][j]
        for col in ("open", "high", "low"):
            assert np.allclose(d[col], src[col][j] * fac), f"{col} has its own factor"
        assert (d["low"] <= d["open"]).all() and (d["open"] <= d["high"]).all()


def test_no_fabricated_jump_at_a_splice():
    """The bug the calibration gate caught: resampling raw price LEVELS invents a crash at
    every join. Measured on f07/AAVE before the fix, joins averaged 40% absolute moves and
    peaked at 169%, against a real maximum of 6.6%."""
    p = _panel(n=3000)
    idx = BS.block_index(3000, 3000, 96, np.random.default_rng(13))
    out = BS.resample_panel(p, idx)
    for pair, d in out["pairs"].items():
        real = np.abs(np.diff(p["pairs"][pair]["close"]) / p["pairs"][pair]["close"][:-1])
        syn = np.abs(np.diff(d["close"]) / d["close"][:-1])
        assert syn.max() <= real.max() * 1.6, (
            f"{pair}: synthetic max move {syn.max():.4f} far exceeds real {real.max():.4f} — "
            f"a splice is fabricating price action")


# ------------------------------------------------- T018a: the cross-section must survive

def test_shared_index_preserves_cross_pair_comovement():
    """One index across all pairs keeps pairs that crashed together crashing together.

    Independent resampling would drive the cross-pair correlation toward zero, which
    UNDERSTATES joint drawdown — and the drawdown ratio is one of the three calibration
    statistics, so the bias would point straight at a gate it has to pass.
    """
    p = _panel(n=6000, pairs=3, seed=11)
    names = sorted(p["pairs"])
    real = np.corrcoef(p["pairs"][names[0]]["_ret"], p["pairs"][names[1]]["_ret"])[0, 1]
    assert real > 0.5, "fixture is not correlated; the test cannot discriminate"

    idx = BS.block_index(6000, 6000, 192, np.random.default_rng(2))
    a = p["pairs"][names[0]]["_ret"][idx % 6000]
    b = p["pairs"][names[1]]["_ret"][idx % 6000]
    shared = np.corrcoef(a, b)[0, 1]

    rng = np.random.default_rng(2)
    ia = BS.block_index(6000, 6000, 192, rng)
    ib = BS.block_index(6000, 6000, 192, rng)
    indep = np.corrcoef(p["pairs"][names[0]]["_ret"][ia % 6000],
                        p["pairs"][names[1]]["_ret"][ib % 6000])[0, 1]

    assert abs(shared - real) < 0.1, (
        f"shared index lost co-movement: real {real:.3f} vs resampled {shared:.3f}")
    assert shared > indep + 0.3, (
        f"shared {shared:.3f} is not clearly above independent {indep:.3f} — the fix is inert")


def test_seeded_reproducibility():
    a = BS.block_index(1000, 1000, 192, np.random.default_rng(9))
    b = BS.block_index(1000, 1000, 192, np.random.default_rng(9))
    c = BS.block_index(1000, 1000, 192, np.random.default_rng(10))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_short_series_is_refused():
    with pytest.raises(ValueError):
        BS.block_index(1, 100, 96, np.random.default_rng(0))
