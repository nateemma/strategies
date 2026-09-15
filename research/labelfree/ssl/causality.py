"""Causality by deletion, plus a planted leak (T017-T020).

THE ARCHITECTURE'S GUARANTEE IS NOT EVIDENCE. Every convolution in ts2vec.py is left-padded, so
output t provably depends only on inputs <= t. That is an argument. This module measures it.

TWO CHECKS, and the second is the one that matters:

  deletion      recompute a bar's embedding with every later bar removed; require equality
  planted leak  inject the forward return as an extra input channel and require the screen's
                rho to JUMP

A causality test that cannot detect a leak it was HANDED is not evidence — it would pass just as
happily on a broken encoder. Feature 011 used exactly this pair, and the planted leak is what
turns "the test passed" into "the test can fail".

A screen run on an uncausal embedding is VOID, not negative: a null from a leaky encoder says
nothing about the hypothesis.
"""
from __future__ import annotations

import numpy as np

from research.labelfree.ssl import ts2vec as TV

TOL = 1e-4          # MLX float32; bit-equality is not available across differing graph shapes


def deletion_test(model, X: np.ndarray, probes: int = 8, seed: int = 0) -> dict:
    """A bar's embedding must not change when every LATER bar is deleted."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.integers(max(600, n // 4), n, size=probes)
    full = TV.embed(model, X)
    worst, rows = 0.0, []
    for t in sorted(set(int(i) for i in idx)):
        trunc = TV.embed(model, X[:t + 1])          # nothing after t exists at all
        d = float(np.max(np.abs(full[t] - trunc[t])))
        worst = max(worst, d)
        rows.append({"bar": t, "max_abs_diff": round(d, 10)})
    return {"probes": rows, "max_abs_diff": worst, "tol": TOL, "passed": worst <= TOL}


def planted_leak(X: np.ndarray, fwd: np.ndarray, seed: int = 7,
                 epochs: int = 10) -> dict:
    """Hand the encoder the answer and require the screen to notice.

    Appends the forward return as an extra input channel. If ridge-on-embedding rho does not
    rise materially, the screen cannot see information that is definitionally present, and no
    null it produces can be trusted.

    THE LEAK MUST BE STRUCTURED, NOT IID NOISE, and getting that wrong cost a false alarm during
    implementation. A contrastive objective optimises representation consistency, so it
    legitimately DISCARDS a channel that is pure noise — measured: 7% of an iid leak survives the
    encoder, against 97-105% of a structured one. An iid leak therefore fails this test on a
    perfectly good encoder.

    A real forward return is not iid: over an H-bar horizon, adjacent bars overlap by H-1 bars,
    so the series is strongly autocorrelated. Callers must pass a REAL forward return (or a
    realistically smooth surrogate); `is_structured` below records whether they did, so a future
    failure can be attributed rather than guessed at.
    """
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr

    n = min(len(X), len(fwd))
    X, fwd = X[:n], fwd[:n]

    def rho_of(Xin):
        m = TV.fit(Xin, seed=seed, epochs=epochs)
        z = TV.embed(m, Xin)
        cut = int(0.6 * len(z))
        r = Ridge(alpha=1.0).fit(z[:cut], fwd[:cut])
        v = spearmanr(r.predict(z[cut:]), fwd[cut:]).statistic
        return float(v) if np.isfinite(v) else 0.0

    # Lag-1 autocorrelation of the planted channel. A real forward return over H bars is
    # strongly autocorrelated because adjacent windows overlap; an iid channel is not, and a
    # contrastive encoder will correctly discard it.
    ac = float(np.corrcoef(fwd[:-1], fwd[1:])[0, 1]) if len(fwd) > 2 else 0.0

    clean = rho_of(X)
    leaked = rho_of(np.column_stack([X, fwd.astype(np.float32)]))
    return {"rho_clean": round(clean, 4), "rho_leaked": round(leaked, 4),
            "jump": round(leaked - clean, 4),
            "leak_autocorr": round(ac, 4),
            "is_structured": bool(ac > 0.5),
            "detected": bool(leaked > max(0.10, abs(clean) + 0.05)),
            "note": ("a screen that cannot see a leak it was handed cannot be trusted to "
                     "report a null; but the leak must be STRUCTURED — a contrastive encoder "
                     "correctly discards iid noise (7% retained vs 97-105% for structure)")}
