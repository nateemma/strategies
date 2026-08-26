"""Manifold-adherence metrics — the block generic fidelity metrics miss.

MT_DDPM's defect (GAN_TODO §5) was synth whose z-scores SATURATED the ±4σ clip:
σ_syn ≈ 4-5× σ_real, OFF_DIST on all 18 buckets. Mean/std RMSE and correlation
deltas can look tolerable while that is happening, so the quality suite would not
have caught it. These two metrics would.

Separately, GAN_TODO §5 Phase 2 found that widening dispersion post-hoc preserved
correlations EXACTLY yet still failed, because the widened points sat OFF the
nonlinear manifold (the AE rejected ~98%). ``nn_distance_ratio`` is the cheap
detector for that: it is insensitive to marginals and correlations, and measures
only whether synth points lie where real points lie.
"""

from __future__ import annotations

import numpy as np


def _flat(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(x.shape[0], -1) if x.ndim > 2 else x


def clip_band_fraction(synth_z: np.ndarray, clip: float = 4.0,
                       tol: float = 1e-6) -> float:
    """Fraction of z-scored synth values pinned at ±clip.

    Near 0 is healthy. A large value means the sampler is producing values the
    clip is truncating -- the generator is not fitting the data range, and the
    clip is masking it rather than fixing it.
    """
    z = _flat(synth_z)
    if z.size == 0:
        return float("nan")
    return float((np.abs(np.abs(z) - clip) <= tol).mean())


def nn_distance_ratio(real: np.ndarray, synth: np.ndarray,
                      *, max_n: int = 2000, seed: int = 0) -> float:
    """median NN distance (synth->real) / median NN distance (real->real).

    ~1.0  synth sits among the real points (on-manifold)
    >>1.0 synth sits off the data manifold, even if its moments match
    <<1.0 synth is collapsed onto / memorising real points

    Standardised per feature first so the ratio is scale-free.
    """
    r, s = _flat(real), _flat(synth)
    if r.size == 0 or s.size == 0 or r.shape[1] != s.shape[1]:
        return float("nan")

    mu, sd = r.mean(0), r.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    r = (r - mu) / sd
    s = (s - mu) / sd
    if not (np.isfinite(r).all() and np.isfinite(s).all()):
        return float("nan")

    rng = np.random.default_rng(seed)
    if len(r) > max_n:
        r = r[rng.choice(len(r), max_n, replace=False)]
    if len(s) > max_n:
        s = s[rng.choice(len(s), max_n, replace=False)]
    if len(r) < 2:
        return float("nan")

    def _median_nn(a, b, exclude_self):
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        if exclude_self:
            np.fill_diagonal(d, np.inf)
        return float(np.median(d.min(axis=1)))

    base = _median_nn(r, r, True)
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    return _median_nn(s, r, False) / base
