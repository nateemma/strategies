"""Density-based rejection sampling for synthetic GAN/DDPM samples.

Fits a per-class density model (Gaussian mixture) on the real
same-class pool, scores each synthetic sample by log-likelihood, and
keeps the top fraction. The lowest-density samples are the ones that
fell in tails or off-manifold regions the GAN didn't model well —
removing them stops the classifier from learning those imperfect
patterns.

Used by ``balance_single_task`` (and balance_multi_task) between
generation and the passthrough swap. Falls back to passing samples
through unchanged when the density model can't be fit safely (too
few real samples, scoring failure, etc.) — failure is non-fatal.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def density_inflate_factor(reject_pct: float) -> float:
    """Multiplier for the requested sample count so post-rejection
    output meets the original target. e.g. reject_pct=0.20 → 1.25."""
    if reject_pct <= 0.0:
        return 1.0
    if reject_pct >= 1.0:
        return 1.0  # nonsensical; caller should disable
    return 1.0 / (1.0 - reject_pct)


def fit_density_model(
    real_pool: Union[np.ndarray, pd.DataFrame],
    n_components: int = 8,
    random_state: int = 0,
) -> Optional[GaussianMixture]:
    """Fit a diagonal-covariance GMM on the real pool. Returns None when
    the pool is too small to fit reliably.

    Diagonal covariance is the conservative choice for low-sample
    regimes — full covariance needs O(F²) samples per component to
    estimate reliably, and we're often working with ~1000-10000 rows.
    """
    arr = (
        real_pool.to_numpy() if isinstance(real_pool, pd.DataFrame)
        else np.asarray(real_pool)
    )
    # Need at least n_components × 2 rows for a remotely meaningful fit.
    if len(arr) < n_components * 2:
        return None
    if not np.all(np.isfinite(arr)):
        # NaN/Inf rows would poison the fit — fall back rather than mask.
        return None
    try:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            random_state=random_state,
            max_iter=100,
            reg_covar=1e-4,
        )
        gmm.fit(arr)
        return gmm
    except Exception:
        return None


def filter_by_density(
    synth: Union[np.ndarray, pd.DataFrame],
    real_pool: Union[np.ndarray, pd.DataFrame],
    reject_pct: float,
    n_components: int = 8,
) -> Union[np.ndarray, pd.DataFrame]:
    """Score synth against real_pool, drop the lowest-density fraction.

    Returns synth unchanged if reject_pct ≤ 0, the density model fails
    to fit, scoring raises, or every sample would be kept anyway.
    Preserves the input type (ndarray or DataFrame).
    """
    if reject_pct <= 0.0:
        return synth
    gmm = fit_density_model(real_pool, n_components=n_components)
    if gmm is None:
        return synth

    synth_arr = (
        synth.to_numpy() if isinstance(synth, pd.DataFrame)
        else np.asarray(synth)
    )
    if not np.all(np.isfinite(synth_arr)):
        return synth
    try:
        scores = gmm.score_samples(synth_arr)
    except Exception:
        return synth

    n_total = len(synth_arr)
    n_keep = max(1, int(n_total * (1.0 - reject_pct)))
    if n_keep >= n_total:
        return synth

    # Top-k by score, preserving original order so passthrough alignment
    # downstream isn't affected by sorting.
    keep_idx = np.sort(np.argsort(scores)[-n_keep:])
    if isinstance(synth, pd.DataFrame):
        return synth.iloc[keep_idx].reset_index(drop=True)
    return synth_arr[keep_idx]
