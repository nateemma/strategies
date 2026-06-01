"""Mahalanobis-distance rejection sampling for synthetic GAN samples.

Scores each synth row by its Mahalanobis distance from the real same-
class distribution:

    d²(x) = (x − μ_real)ᵀ Σ_real⁻¹ (x − μ_real)

Unlike the realsignal classifier (which scores class-membership
confidence and rewards prototype-like samples), and unlike the diagonal-
cov density filter (which only sees marginal density), Mahalanobis with
the **full** covariance directly addresses the joint-distribution drift
metric (|Δρ|) reported in the augmentation diagnostics.

The covariance is fit with Ledoit-Wolf shrinkage so it's well-
conditioned for inversion even when the per-class pool is small or
features are correlated.

Two modes (mirroring the discriminator filters):
  * Rank-based (``filter_by_mahalanobis``): drop the top fraction by
    distance — keep the rows closest to the real centroid.
  * Threshold-based (``filter_by_mahalanobis_threshold``): keep rows
    with d² below the threshold. The threshold operates on raw squared
    Mahalanobis distance; under multivariate normality this is
    chi-squared distributed with F degrees of freedom, so a sensible
    threshold for F=24 features is in the [20, 50] range (50 ≈ 99.8th
    percentile under MVN).

Used by ``balance_single_task`` between generation and the passthrough
swap. Falls through (returns synth unchanged) when the real pool is too
small, fitting fails, NaN/Inf, etc. — failures are non-fatal.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def mahalanobis_inflate_factor(reject_pct: float) -> float:
    """Multiplier for generated count so post-rejection output meets the
    original target."""
    if reject_pct <= 0.0:
        return 1.0
    if reject_pct >= 1.0:
        return 1.0
    return 1.0 / (1.0 - reject_pct)


def _fit_real_distribution(real_pool: Union[np.ndarray, pd.DataFrame]):
    """Fit Ledoit-Wolf shrinkage covariance + mean on real_pool. Returns
    a fitted estimator with `.mean_`, `.covariance_`, `.mahalanobis(X)`.
    Returns None on failure."""
    arr = (
        real_pool.to_numpy() if isinstance(real_pool, pd.DataFrame)
        else np.asarray(real_pool)
    )
    if arr.ndim != 2:
        return None
    n_samples, n_features = arr.shape
    # Need at least F+1 samples for a remotely meaningful covariance;
    # shrinkage handles ill-conditioning but garbage-in-garbage-out.
    if n_samples < max(n_features + 1, 30):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    try:
        est = LedoitWolf()
        est.fit(arr)
        return est
    except Exception:
        return None


def _score_distances(
    est, synth: Union[np.ndarray, pd.DataFrame]
) -> Union[np.ndarray, None]:
    """Compute squared Mahalanobis distances for synth rows under est.
    Returns None on failure."""
    arr = (
        synth.to_numpy() if isinstance(synth, pd.DataFrame)
        else np.asarray(synth)
    )
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        return None
    try:
        return est.mahalanobis(arr).astype(np.float32)
    except Exception:
        return None


def filter_by_mahalanobis(
    synth: Union[np.ndarray, pd.DataFrame],
    real_pool: Union[np.ndarray, pd.DataFrame],
    reject_pct: float,
) -> Union[np.ndarray, pd.DataFrame]:
    """Drop the top ``reject_pct`` fraction of synth rows by Mahalanobis
    distance to the real same-class distribution. Returns synth
    unchanged on any failure or when ``reject_pct <= 0``.
    """
    if reject_pct <= 0.0:
        return synth
    est = _fit_real_distribution(real_pool)
    if est is None:
        return synth
    distances = _score_distances(est, synth)
    if distances is None:
        return synth

    n_total = len(distances)
    n_keep = max(1, int(n_total * (1.0 - reject_pct)))
    if n_keep >= n_total:
        return synth

    # Keep the *lowest* distances (closest to real centroid), preserving
    # original row order so downstream passthrough alignment stays valid.
    keep_idx = np.sort(np.argsort(distances)[:n_keep])
    if isinstance(synth, pd.DataFrame):
        return synth.iloc[keep_idx].reset_index(drop=True)
    arr = (
        synth.to_numpy() if isinstance(synth, pd.DataFrame)
        else np.asarray(synth)
    )
    return arr[keep_idx]


def filter_by_mahalanobis_threshold(
    synth: Union[np.ndarray, pd.DataFrame],
    real_pool: Union[np.ndarray, pd.DataFrame],
    threshold: float,
) -> Union[np.ndarray, pd.DataFrame]:
    """Keep only synth rows with squared Mahalanobis distance below
    ``threshold``. Returns synth unchanged on any failure. Variable
    output count.
    """
    est = _fit_real_distribution(real_pool)
    if est is None:
        return synth
    distances = _score_distances(est, synth)
    if distances is None:
        return synth

    keep_mask = distances < float(threshold)
    if not keep_mask.any():
        # Return an empty container so the caller's iteration / label
        # build sees a zero-row case rather than None.
        if isinstance(synth, pd.DataFrame):
            return synth.iloc[0:0].reset_index(drop=True)
        arr = np.asarray(synth)
        return arr[:0]

    if isinstance(synth, pd.DataFrame):
        return synth.iloc[keep_mask].reset_index(drop=True)
    arr = np.asarray(synth)
    return arr[keep_mask]
