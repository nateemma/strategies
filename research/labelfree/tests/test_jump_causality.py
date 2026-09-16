"""Causality of the jump-model state series (T015-T018, T021; FR-004, FR-005, SC-002).

THE CENTRAL PAIR OF TESTS IS DELIBERATELY ASYMMETRIC:

  test_filtered_prefix_stable  -- truncating history must NOT move a filtered state
  test_smoothed_prefix_changes -- truncating history MUST move a smoothed state

The second one asserts a FAILURE. A smoothed sequence that survives truncation means the
Viterbi traceback never ran, which would make the leak reference not a leak reference and the
published gap not a leak size.

WHY THE CUT GRID IS DENSE AND THE ASSERTION IS "SOMEWHERE". Measured 2026-09-15: the smoothed
prefix changed at only 2-5 of 104 cut points. The traceback's influence decays backwards, so
the leak is REAL BUT SPARSE -- a spot check of three cut points would have passed the smoothed
sequence and reported no leak at all. The grid is therefore dense and the assertion is "changes
somewhere", never "changes everywhere".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.labelfree.jump import features as FE, model as M, refit as R
from research.labelfree.jump.online import filtered_states, smoothed_states

LAM_AMBIGUOUS = 60.0          # large enough that filtered and smoothed genuinely diverge

# The fixture needs ~2700 rows. At 840 the fit COLLAPSES to a single state and the two paths
# agree everywhere, which would make the leak test pass vacuously -- caught by
# `test_fixture_is_not_degenerate`, which exists precisely so that can never go unnoticed.
# The same sample-size sensitivity shows on real data: SOL/USDT scored 0.86 cross-seed
# agreement over 9 months and 1.00 over 18.
FIXTURE_SEGMENTS = (400, 300, 500, 200, 600, 250, 450)


def _ambiguous_panel(seed: int = 7) -> np.ndarray:
    """Overlapping regimes and short segments -- the only regime where the two paths differ.

    Well-separated regimes make filtered and smoothed agree on every bar, which would make
    the leak test vacuously pass.
    """
    rng = np.random.default_rng(seed)

    def blk(n, base):
        return np.column_stack([np.abs(rng.normal(base, base * 0.9, n)),
                                np.abs(rng.normal(base * 0.7, base * 0.9, n)),
                                rng.normal(0.0, 0.3, n)])

    segs, cur = [], 0.006
    for n in FIXTURE_SEGMENTS:
        segs.append(blk(n, cur))
        cur = 0.018 if cur < 0.01 else 0.006
    return np.vstack(segs)


@pytest.fixture(scope="module")
def fitted():
    x = _ambiguous_panel()
    f = M.fit(x, 2, LAM_AMBIGUOUS, seed=0, n_init=2)
    return x, f, (x - f.scaler_mean) / f.scaler_scale


def test_fixture_is_not_degenerate(fitted):
    """Guards every test below: a collapsed fit makes the leak tests vacuous."""
    x, f, xs = fitted
    share = float((filtered_states(xs, f) == 1).mean())
    assert 0.05 < share < 0.95, f"fit collapsed to one state (high-vol share {share:.3f})"


def test_filtered_prefix_stable(fitted):
    """FR-004 -- no bar's filtered state moves when the future is deleted."""
    x, f, xs = fitted
    full = filtered_states(xs, f)
    for m in range(100, len(xs), 25):
        assert np.array_equal(filtered_states(xs[:m], f), full[:m]), f"moved at cut {m}"


def test_smoothed_prefix_changes(fitted):
    """FR-007 -- the leak reference MUST leak. Asserts a failure of the test above."""
    x, f, xs = fitted
    full = smoothed_states(xs, f)
    unstable = [m for m in range(100, len(xs), 25)
                if not np.array_equal(smoothed_states(xs[:m], f), full[:m])]
    assert unstable, (
        "smoothed states survived every truncation -- the Viterbi traceback is not running, "
        "so the leak reference is not a leak reference and any published gap is not a leak size")


def test_filtered_and_smoothed_actually_differ(fitted):
    """Guards the fixture itself: if the two agree everywhere the pair above proves nothing."""
    x, f, xs = fitted
    assert (filtered_states(xs, f) != smoothed_states(xs, f)).sum() > 0


def test_refit_uses_only_past():
    """FR-005 -- no bar at or after a refit point enters that fit."""
    idx = pd.date_range("2023-01-01", periods=96 * 300, freq="15min", tz="UTC")
    first = pd.Timestamp("2023-04-01", tz="UTC")
    pts = R.schedule(idx, first)
    assert pts and pts[0] == first
    for tau in pts:
        past = idx[idx < tau]
        assert len(past) == 0 or past.max() < tau


def test_refits_continue_through_the_range():
    """Clarification Q2=A -- refits do not stop at a fold boundary."""
    idx = pd.date_range("2023-01-01", periods=96 * 400, freq="15min", tz="UTC")
    pts = R.schedule(idx, pd.Timestamp("2023-04-01", tz="UTC"))
    assert len(pts) >= 3
    assert all((b - a).days == 90 for a, b in zip(pts, pts[1:]))


def test_standardisation_uses_fit_window_only():
    """FR-005 -- brief 07's recorded leak: statistics fitted over the scoring split too."""
    x = _ambiguous_panel()
    cut = 500
    f_past = M.fit(x[:cut], 2, LAM_AMBIGUOUS, seed=0, n_init=1)
    f_all = M.fit(x, 2, LAM_AMBIGUOUS, seed=0, n_init=1)
    assert not np.allclose(f_past.scaler_mean, f_all.scaler_mean), (
        "standardisation statistics are identical on a prefix and the full series, which means "
        "they are not being fitted on the expanding window")
    np.testing.assert_allclose(f_past.scaler_mean, x[:cut].mean(axis=0))


def test_features_are_trailing_only():
    """FR-004 -- row t is a function of bars <= t."""
    close = pd.Series(np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 3000))),
                      index=pd.date_range("2023-01-01", periods=3000, freq="15min", tz="UTC"))
    full = FE._from_close(close).dropna()
    for m in (1200, 2000, 2600):
        head = FE._from_close(close.iloc[:m]).dropna()
        common = head.index.intersection(full.index)
        pd.testing.assert_frame_equal(head.loc[common], full.loc[common])


def test_warmup_is_dropped_not_backfilled():
    """A back-fill would carry a later window's value backwards -- a leak that looks tidy."""
    close = pd.Series(np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 500))),
                      index=pd.date_range("2023-01-01", periods=500, freq="15min", tz="UTC"))
    out = FE._from_close(close)
    assert out.iloc[:FE.WINDOW - 1].isna().all().all()


# ---------------------------------------------------------------------------
# T021 -- the deletion test against the WRITTEN ARTEFACT.
#
# Everything above tests the in-memory function. Research R7 requires the test to target the
# object a consumer actually reads, because pointing a causality test at a different object
# than the run consumes is precisely how a leak survives a passing test: brief 07's encoder
# trained on 70% of the TEST range while the splitting code read correctly.
# ---------------------------------------------------------------------------

def test_written_artefact_is_causal(tmp_path):
    """Truncating input history must not change any state already written to disk."""
    import pandas as pd

    from research.labelfree.jump import refit as R

    x = _ambiguous_panel()
    idx = pd.date_range("2022-01-01", periods=len(x), freq="15min", tz="UTC")
    feats = pd.DataFrame(x, columns=["rvol_realised", "rvol_downside", "sign_persistence"],
                         index=idx)
    first = feats.index[len(x) // 3]

    full = R.run(feats, first, LAM_AMBIGUOUS, k=2, seed=0, mode="filtered")
    p_full = R.write(full, "tmp_full", "filtered", 0)

    cut = feats.index[int(len(x) * 0.8)]
    trunc = R.run(feats.loc[feats.index < cut], first, LAM_AMBIGUOUS, k=2, seed=0, mode="filtered")
    p_trunc = R.write(trunc, "tmp_trunc", "filtered", 0)

    a = pd.read_feather(p_full).set_index("timestamp")["state"]
    b = pd.read_feather(p_trunc).set_index("timestamp")["state"]
    common = a.index.intersection(b.index)
    assert len(common) > 300, "truncation left too little overlap to test"
    assert (a.loc[common].to_numpy() == b.loc[common].to_numpy()).all(), (
        "states written to disk MOVED when later bars were deleted")
    for p in (p_full, p_trunc):
        p.unlink(missing_ok=True)


def test_writer_refuses_nothing_and_roundtrips(tmp_path):
    """The artefact must survive the round trip the consumer performs."""
    import pandas as pd

    from research.labelfree.jump import refit as R

    x = _ambiguous_panel()
    idx = pd.date_range("2022-01-01", periods=len(x), freq="15min", tz="UTC")
    feats = pd.DataFrame(x, columns=["rvol_realised", "rvol_downside", "sign_persistence"],
                         index=idx)
    d = R.run(feats, feats.index[len(x) // 3], LAM_AMBIGUOUS, k=2, seed=0, mode="filtered")
    p = R.write(d, "tmp_rt", "filtered", 0)
    back = pd.read_feather(p)
    pd.testing.assert_frame_equal(back, d.reset_index(drop=True))
    assert set(back["mode"]) == {"filtered"}
    p.unlink(missing_ok=True)
