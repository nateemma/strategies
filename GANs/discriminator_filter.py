"""Discriminator-based rejection sampling for synthetic GAN/DDPM samples.

Two implementations:

* ``filter_by_discriminator`` — in-loop sklearn HistGradientBoostingClassifier.
  Trains a fresh per-class discriminator each balance call. Lightweight
  but capacity-limited (max_depth=4) and trained on the same synth it
  filters.

* ``filter_by_neural_discriminator`` — loads a unified MLX
  RealnessDiscriminator trained ONCE on real-vs-pooled-synth-from-all-GANs
  (via the CreateDiscriminator strategy) and uses it to score per-row
  realness. Cross-label aware (class one-hot input), more capacity,
  and trained on a diverse fakes pool so it generalises across GANs.

Both complement density_filter.py:
  * GMM density filter — fits a diagonal-covariance GMM on real samples
    only, scores by marginal density. Misses joint-correlation drift.
  * Discriminator filters — see both real and synth distributions, so
    they catch joint structure breaks (broken correlations, sign-flipped
    feature pairs) that marginal density can't.

Used by balance_single_task between generation and the passthrough swap.
Both filter functions fall back to passing samples through unchanged on
any failure (model file missing, NaN/Inf, fit/scoring error) — failure
is non-fatal.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def discriminator_inflate_factor(reject_pct: float) -> float:
    """Multiplier for the requested sample count so post-rejection
    output meets the original target."""
    if reject_pct <= 0.0:
        return 1.0
    if reject_pct >= 1.0:
        return 1.0
    return 1.0 / (1.0 - reject_pct)


def filter_by_discriminator(
    synth: Union[np.ndarray, pd.DataFrame],
    real_pool: Union[np.ndarray, pd.DataFrame],
    reject_pct: float,
    max_iter: int = 50,
    max_depth: int = 4,
    random_state: int = 0,
) -> Union[np.ndarray, pd.DataFrame]:
    """Train a discriminator on (real, synth), drop the bottom
    reject_pct fraction of synth by P(real).

    Returns synth unchanged if reject_pct ≤ 0, either pool is too small
    to train, NaN/Inf inputs, the classifier fails to fit, or every
    sample would be kept anyway.

    Preserves the input type (ndarray or DataFrame). Output order
    matches the original input row order (sorted-by-keep-index) so
    downstream passthrough alignment is preserved.
    """
    if reject_pct <= 0.0:
        return synth

    synth_arr = (
        synth.to_numpy() if isinstance(synth, pd.DataFrame)
        else np.asarray(synth)
    )
    real_arr = (
        real_pool.to_numpy() if isinstance(real_pool, pd.DataFrame)
        else np.asarray(real_pool)
    )

    # Need enough rows on both sides for a meaningful split. The
    # discriminator's early-stopping validation needs a few hundred
    # samples to be useful; below that, fall through.
    if len(real_arr) < 100 or len(synth_arr) < 50:
        return synth
    if not (np.all(np.isfinite(synth_arr)) and np.all(np.isfinite(real_arr))):
        return synth

    n_total = len(synth_arr)
    n_keep = max(1, int(n_total * (1.0 - reject_pct)))
    if n_keep >= n_total:
        return synth

    X = np.vstack([real_arr, synth_arr])
    y = np.concatenate([
        np.zeros(len(real_arr), dtype=np.int8),
        np.ones(len(synth_arr), dtype=np.int8),
    ])

    try:
        clf = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            random_state=random_state,
        )
        clf.fit(X, y)
        p_synth = clf.predict_proba(synth_arr)[:, 1]
    except Exception:
        return synth

    # Realness score = 1 - P(synth). Higher = more real-looking.
    realness = 1.0 - p_synth

    # Top-k by realness, preserving original order so downstream
    # passthrough alignment isn't affected by sorting.
    keep_idx = np.sort(np.argsort(realness)[-n_keep:])
    if isinstance(synth, pd.DataFrame):
        return synth.iloc[keep_idx].reset_index(drop=True)
    return synth_arr[keep_idx]


# ---------------------------------------------------------------------------
# Neural-discriminator filter (load saved RealnessDiscriminator + score)
# ---------------------------------------------------------------------------

# Module-level cache so the model isn't reloaded per-class per-pair. The
# discriminator is trained once and re-used across every balance call in
# the backtest. Keyed by absolute model path so a fresh strategy that
# changes the path doesn't get a stale cache hit.
_NEURAL_MODEL_CACHE: dict = {}


def _load_neural_discriminator(model_path: str):
    """Best-effort load + cache. Returns None on any failure so callers
    can fall through."""
    abs_path = os.path.abspath(model_path)
    if abs_path in _NEURAL_MODEL_CACHE:
        return _NEURAL_MODEL_CACHE[abs_path]
    try:
        from Discriminators.RealnessDiscriminator import RealnessDiscriminator
    except Exception:
        _NEURAL_MODEL_CACHE[abs_path] = None
        return None
    try:
        model = RealnessDiscriminator.load(abs_path)
    except Exception:
        model = None
    _NEURAL_MODEL_CACHE[abs_path] = model
    return model


def filter_by_neural_discriminator(
    synth: Union[np.ndarray, pd.DataFrame],
    class_idx: int,
    num_classes: int,
    reject_pct: float,
    model_path: str,
) -> Union[np.ndarray, pd.DataFrame]:
    """Score synth via the saved unified RealnessDiscriminator and drop
    the bottom reject_pct fraction by P(real).

    Returns synth unchanged if reject_pct ≤ 0, the model can't be
    loaded, NaN/Inf inputs, scoring fails, or every sample would be
    kept anyway.

    Preserves the input type (ndarray or DataFrame). Output order
    matches the original input row order so downstream passthrough
    alignment is preserved.
    """
    if reject_pct <= 0.0:
        return synth

    model = _load_neural_discriminator(model_path)
    if model is None:
        return synth

    synth_arr = (
        synth.to_numpy() if isinstance(synth, pd.DataFrame)
        else np.asarray(synth)
    )
    if synth_arr.ndim != 2 or not np.all(np.isfinite(synth_arr)):
        return synth
    if synth_arr.shape[1] != model.num_features:
        return synth
    if not (0 <= int(class_idx) < int(num_classes)):
        return synth

    n_total = len(synth_arr)
    n_keep = max(1, int(n_total * (1.0 - reject_pct)))
    if n_keep >= n_total:
        return synth

    try:
        import mlx.core as mx
        from Discriminators.RealnessDiscriminator import one_hot_class
        x_mx = mx.array(synth_arr.astype(np.float32))
        c_idx = np.full(n_total, int(class_idx), dtype=np.int64)
        c_mx = one_hot_class(c_idx, num_classes=int(num_classes))
        scores_mx = model.score(x_mx, c_mx)
        realness = np.asarray(scores_mx, dtype=np.float32).flatten()
    except Exception:
        return synth

    keep_idx = np.sort(np.argsort(realness)[-n_keep:])
    if isinstance(synth, pd.DataFrame):
        return synth.iloc[keep_idx].reset_index(drop=True)
    return synth_arr[keep_idx]


# ---------------------------------------------------------------------------
# Realsignal filter (per-class binary classifier trained on real only)
# ---------------------------------------------------------------------------

# Cache realsignal models per (model_root, class_idx). Loading is lazy
# but happens at most once per (root, class) tuple over the whole
# backtest. Keys use the absolute root path so changing roots doesn't
# get a stale hit.
_REALSIGNAL_MODEL_CACHE: dict = {}


def _load_realsignal_classifier(model_root: str, class_idx: int):
    """Best-effort load + cache for a per-class binary signal model.
    Returns None on any failure so callers can fall through."""
    abs_root = os.path.abspath(model_root)
    key = (abs_root, int(class_idx))
    if key in _REALSIGNAL_MODEL_CACHE:
        return _REALSIGNAL_MODEL_CACHE[key]
    try:
        from Discriminators.RealsignalClassifier import (
            RealsignalClassifier,
            class_subdir,
        )
    except Exception:
        _REALSIGNAL_MODEL_CACHE[key] = None
        return None
    try:
        path = class_subdir(abs_root, int(class_idx))
        model = RealsignalClassifier.load(path)
    except Exception:
        model = None
    _REALSIGNAL_MODEL_CACHE[key] = model
    return model


def filter_by_realsignal(
    synth: Union[np.ndarray, pd.DataFrame],
    class_idx: int,
    reject_pct: float,
    model_root: str,
) -> Union[np.ndarray, pd.DataFrame]:
    """Score synth via the saved per-class RealsignalClassifier and drop
    the bottom reject_pct fraction by P(real signal of class c).

    Returns synth unchanged if reject_pct ≤ 0, no model for the
    requested class exists at ``model_root``, NaN/Inf inputs, scoring
    fails, or every sample would be kept anyway.
    """
    if reject_pct <= 0.0:
        return synth

    model = _load_realsignal_classifier(model_root, int(class_idx))
    if model is None:
        return synth

    synth_arr = (
        synth.to_numpy() if isinstance(synth, pd.DataFrame)
        else np.asarray(synth)
    )
    if synth_arr.ndim != 2 or not np.all(np.isfinite(synth_arr)):
        return synth
    if synth_arr.shape[1] != model.num_features:
        return synth

    n_total = len(synth_arr)
    n_keep = max(1, int(n_total * (1.0 - reject_pct)))
    if n_keep >= n_total:
        return synth

    try:
        import mlx.core as mx
        x_mx = mx.array(synth_arr.astype(np.float32))
        scores_mx = model.score(x_mx)
        signal_score = np.asarray(scores_mx, dtype=np.float32).flatten()
    except Exception:
        return synth

    keep_idx = np.sort(np.argsort(signal_score)[-n_keep:])
    if isinstance(synth, pd.DataFrame):
        return synth.iloc[keep_idx].reset_index(drop=True)
    return synth_arr[keep_idx]


# ---------------------------------------------------------------------------
# Autoencoder filter (per-class manifold reconstruction error)
# ---------------------------------------------------------------------------

_AUTOENCODER_MODEL_CACHE: dict = {}


def _load_autoencoder(model_root: str, class_idx: int):
    """Best-effort load + cache for a per-class autoencoder. Returns
    None on any failure so callers can fall through."""
    abs_root = os.path.abspath(model_root)
    key = (abs_root, int(class_idx))
    if key in _AUTOENCODER_MODEL_CACHE:
        return _AUTOENCODER_MODEL_CACHE[key]
    try:
        from Discriminators.RealAutoencoder import (
            RealAutoencoder,
            class_subdir,
        )
    except Exception:
        _AUTOENCODER_MODEL_CACHE[key] = None
        return None
    try:
        path = class_subdir(abs_root, int(class_idx))
        model = RealAutoencoder.load(path)
    except Exception:
        model = None
    _AUTOENCODER_MODEL_CACHE[key] = model
    return model


def _score_autoencoder_mse(model, synth_arr: np.ndarray) -> Optional[np.ndarray]:
    """Compute per-row reconstruction MSE using the saved autoencoder.
    Returns None on any failure."""
    try:
        import mlx.core as mx
        x_mx = mx.array(synth_arr.astype(np.float32))
        mse_mx = model.reconstruction_error(x_mx)
        return np.asarray(mse_mx, dtype=np.float32).flatten()
    except Exception:
        return None


def filter_by_autoencoder(
    synth: Union[np.ndarray, pd.DataFrame],
    class_idx: int,
    reject_pct: float,
    model_root: str,
) -> Union[np.ndarray, pd.DataFrame]:
    """Score synth via the saved per-class autoencoder and drop the top
    ``reject_pct`` fraction by reconstruction MSE (highest MSE =
    most off-manifold = drop).

    Returns synth unchanged if reject_pct ≤ 0, the model for the
    requested class can't be loaded, NaN/Inf inputs, scoring fails,
    or every sample would be kept anyway.
    """
    if reject_pct <= 0.0:
        return synth
    model = _load_autoencoder(model_root, int(class_idx))
    if model is None:
        return synth

    synth_arr = (
        synth.to_numpy() if isinstance(synth, pd.DataFrame)
        else np.asarray(synth)
    )
    if synth_arr.ndim != 2 or not np.all(np.isfinite(synth_arr)):
        return synth
    if synth_arr.shape[1] != model.num_features:
        return synth

    mse = _score_autoencoder_mse(model, synth_arr)
    if mse is None:
        return synth

    n_total = len(synth_arr)
    n_keep = max(1, int(n_total * (1.0 - reject_pct)))
    if n_keep >= n_total:
        return synth

    # Keep the LOWEST MSE (most on-manifold). Preserve original order.
    keep_idx = np.sort(np.argsort(mse)[:n_keep])
    if isinstance(synth, pd.DataFrame):
        return synth.iloc[keep_idx].reset_index(drop=True)
    return synth_arr[keep_idx]


def filter_by_autoencoder_threshold(
    synth: Union[np.ndarray, pd.DataFrame],
    class_idx: int,
    threshold: float,
    model_root: str,
    *,
    return_kept_mask: bool = False,
) -> Union[
    np.ndarray,
    pd.DataFrame,
    "tuple[Union[np.ndarray, pd.DataFrame], Optional[np.ndarray]]",
]:
    """Keep only synth rows with reconstruction MSE below ``threshold``.
    Variable output count.

    When ``return_kept_mask=True``, returns ``(filtered_synth, keep_mask)``
    where ``keep_mask`` is a boolean ndarray of length ``len(synth)`` aligned
    with the input rows, or ``None`` if the model is unavailable or input
    fails validation (in which case ``filtered_synth`` is the original
    ``synth`` unchanged). The mask lets callers synchronise external arrays
    (e.g. multi-task label vectors) with the filtered rows.
    """
    model = _load_autoencoder(model_root, int(class_idx))
    if model is None:
        return (synth, None) if return_kept_mask else synth

    synth_arr = (
        synth.to_numpy() if isinstance(synth, pd.DataFrame)
        else np.asarray(synth)
    )
    if synth_arr.ndim != 2 or not np.all(np.isfinite(synth_arr)):
        return (synth, None) if return_kept_mask else synth
    if synth_arr.shape[1] != model.num_features:
        return (synth, None) if return_kept_mask else synth

    mse = _score_autoencoder_mse(model, synth_arr)
    if mse is None:
        return (synth, None) if return_kept_mask else synth

    keep_mask = mse < float(threshold)
    if not keep_mask.any():
        if isinstance(synth, pd.DataFrame):
            empty = synth.iloc[0:0].reset_index(drop=True)
        else:
            empty = synth_arr[:0]
        return (empty, keep_mask) if return_kept_mask else empty

    if isinstance(synth, pd.DataFrame):
        kept = synth.iloc[keep_mask].reset_index(drop=True)
    else:
        kept = synth_arr[keep_mask]
    return (kept, keep_mask) if return_kept_mask else kept


# ---------------------------------------------------------------------------
# Threshold-based filtering (variable output count)
# ---------------------------------------------------------------------------

def filter_by_realsignal_threshold(
    synth: Union[np.ndarray, pd.DataFrame],
    class_idx: int,
    threshold: float,
    model_root: str,
) -> Union[np.ndarray, pd.DataFrame]:
    """Keep only synth where the per-class RealsignalClassifier scores
    ``P(real class c) > threshold``. Variable output count — could be
    fewer than the input or even zero.

    Falls through (returns synth unchanged) when the model is missing,
    inputs are non-finite, or scoring fails.
    """
    model = _load_realsignal_classifier(model_root, int(class_idx))
    if model is None:
        return synth

    synth_arr = (
        synth.to_numpy() if isinstance(synth, pd.DataFrame)
        else np.asarray(synth)
    )
    if synth_arr.ndim != 2 or not np.all(np.isfinite(synth_arr)):
        return synth
    if synth_arr.shape[1] != model.num_features:
        return synth

    try:
        import mlx.core as mx
        x_mx = mx.array(synth_arr.astype(np.float32))
        scores_mx = model.score(x_mx)
        signal_score = np.asarray(scores_mx, dtype=np.float32).flatten()
    except Exception:
        return synth

    keep_mask = signal_score > float(threshold)
    if not keep_mask.any():
        # All synth scored below threshold. Return an empty container of
        # the same type so the downstream concat/label-build sees a
        # zero-row case rather than a None.
        if isinstance(synth, pd.DataFrame):
            return synth.iloc[0:0].reset_index(drop=True)
        return synth_arr[:0]

    if isinstance(synth, pd.DataFrame):
        return synth.iloc[keep_mask].reset_index(drop=True)
    return synth_arr[keep_mask]


def filter_by_neural_discriminator_threshold(
    synth: Union[np.ndarray, pd.DataFrame],
    class_idx: int,
    num_classes: int,
    threshold: float,
    model_path: str,
) -> Union[np.ndarray, pd.DataFrame]:
    """Keep only synth where the unified RealnessDiscriminator scores
    ``P(real) > threshold``. Variable output count."""
    model = _load_neural_discriminator(model_path)
    if model is None:
        return synth

    synth_arr = (
        synth.to_numpy() if isinstance(synth, pd.DataFrame)
        else np.asarray(synth)
    )
    if synth_arr.ndim != 2 or not np.all(np.isfinite(synth_arr)):
        return synth
    if synth_arr.shape[1] != model.num_features:
        return synth
    if not (0 <= int(class_idx) < int(num_classes)):
        return synth

    try:
        import mlx.core as mx
        from Discriminators.RealnessDiscriminator import one_hot_class
        x_mx = mx.array(synth_arr.astype(np.float32))
        c_idx_arr = np.full(len(synth_arr), int(class_idx), dtype=np.int64)
        c_mx = one_hot_class(c_idx_arr, num_classes=int(num_classes))
        scores_mx = model.score(x_mx, c_mx)
        realness = np.asarray(scores_mx, dtype=np.float32).flatten()
    except Exception:
        return synth

    keep_mask = realness > float(threshold)
    if not keep_mask.any():
        if isinstance(synth, pd.DataFrame):
            return synth.iloc[0:0].reset_index(drop=True)
        return synth_arr[:0]

    if isinstance(synth, pd.DataFrame):
        return synth.iloc[keep_mask].reset_index(drop=True)
    return synth_arr[keep_mask]
