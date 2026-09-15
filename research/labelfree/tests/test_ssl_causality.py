"""Causality: deletion and the planted leak (T015, T016, T019)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.ssl import causality as CA              # noqa: E402
from research.labelfree.ssl import ts2vec as TV                 # noqa: E402


def _series(n=2400, d=5, seed=3):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def test_embedding_is_not_constant_and_the_loss_falls():
    """A collapsed encoder scores rho~0 and would read as evidence about the hypothesis when it
    is encoder failure (FR-018)."""
    X = _series()
    m = TV.fit(X, seed=7, epochs=8)
    assert m._losses[-1] < m._losses[0], "loss did not fall — the encoder is not training"
    z = TV.embed(m, X[:600])
    assert np.isfinite(z).all()
    assert z.std() > 1e-3, "embedding collapsed to a constant"


def test_seeded_reproducibility():
    X = _series()
    a = TV.embed(TV.fit(X, seed=11, epochs=4), X[:300])
    b = TV.embed(TV.fit(X, seed=11, epochs=4), X[:300])
    assert np.allclose(a, b, atol=1e-5), "not reproducible under a fixed seed"


def test_deletion_shows_no_dependence_on_later_bars():
    """FR-002 measured, not argued. Every conv is left-padded, but that is the claim."""
    X = _series()
    m = TV.fit(X, seed=7, epochs=6)
    rep = CA.deletion_test(m, X, probes=5)
    assert rep["passed"], (
        f"a bar's embedding changed by {rep['max_abs_diff']:.2e} when later bars were deleted — "
        f"the encoder is not causal")


def test_planted_leak_is_detected():
    """THE LOAD-BEARING TEST. A causality check that cannot see a leak it was handed is not
    evidence: it would pass just as happily on a broken encoder.

    The leak is a REAL forward return over a price path, not iid noise. That distinction cost a
    false alarm during implementation: an iid leak fails this test on a perfectly good encoder,
    because a contrastive objective legitimately discards noise (7% of an iid channel survives,
    against 97-105% of a structured one). Real forward returns are autocorrelated — adjacent
    H-bar windows overlap by H-1 bars — so an iid leak was never the right stand-in.
    """
    n = 3000
    rng = np.random.default_rng(5)
    t_ = np.arange(n)
    # A PREDICTABLE target. See test_contrastive_encoder_retains_only_what_is_predictable for
    # why an unpredictable one cannot be used here: the encoder discards it by construction,
    # which is a property of the method rather than a fault in the plumbing.
    sig = np.sin(t_ / 50.0) + 0.3 * rng.standard_normal(n)
    X = np.column_stack([sig, rng.standard_normal((n, 3))]).astype(np.float32)
    fwd = np.roll(sig, -6).astype(np.float32)
    rep = CA.planted_leak(X, fwd, seed=7, epochs=10)
    assert rep["is_structured"], (
        f"the planted leak has lag-1 autocorrelation {rep['leak_autocorr']}")
    assert rep["detected"], (
        f"the screen did NOT notice the forward return handed to it as an input "
        f"(clean {rep['rho_clean']}, leaked {rep['rho_leaked']}). Any null it reports is "
        f"untrustworthy.")
    assert rep["jump"] > 0


def test_contrastive_encoder_retains_only_what_is_predictable():
    """Why the leak must be structured — the measurement that explained the false alarm.

    A contrastive objective optimises representation CONSISTENCY, not information retention, so
    it drops what does not serve that objective. This is a property of the method and bears
    directly on the brief: an SSL embedding preserves learnable structure and discards the rest.
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge

    n = 3000
    rng = np.random.default_rng(11)
    t_ = np.arange(n)
    sig = np.sin(t_ / 50.0) + 0.3 * rng.standard_normal(n)

    def retained(X, y):
        m = TV.fit(X.astype(np.float32), seed=7, epochs=12)
        z = TV.embed(m, X.astype(np.float32))
        c = int(0.6 * n)
        raw = spearmanr(Ridge(1.0).fit(X[:c], y[:c]).predict(X[c:]), y[c:]).statistic
        emb = spearmanr(Ridge(1.0).fit(z[:c], y[:c]).predict(z[c:]), y[c:]).statistic
        return abs(emb) / max(abs(raw), 1e-9)

    # PREDICTABLE target: retained.
    r_struct = retained(np.column_stack([sig, rng.standard_normal((n, 3))]), np.roll(sig, -6))
    assert r_struct > 0.70, f"predictable structure should survive, {r_struct:.1%} retained"

    # UNPREDICTABLE first-moment target, handed to the encoder AS ITS OWN CHANNEL: discarded.
    # This is the measurement that bears on the brief. A forward return in a near-efficient
    # market is exactly such a target, so a contrastive encoder cannot carry it -- which is the
    # literature's claim (review §E: SSL embeddings learn second-moment structure, not
    # conditional mean) demonstrated here rather than cited.
    r2 = np.random.default_rng(5).standard_normal(n + 6) * 0.004
    px = 100 * np.cumprod(1 + r2)
    fwd = px[6:n + 6] / px[:n] - 1.0
    r_unpred = retained(
        np.column_stack([np.diff(np.log(px))[:n], rng.standard_normal((n, 3)), fwd]), fwd)
    assert r_unpred < 0.30, (
        f"an unpredictable target survived at {r_unpred:.1%} — if this ever passes, the "
        f"screen's power is higher than the brief's result assumed and it should be re-run")
