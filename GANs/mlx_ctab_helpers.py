"""
Shared helpers for the MLX CTAB-GAN+ family.

Three sections, all pure numpy / pandas / scipy / sklearn — no MLX or TF
imports — so the same code drives both the single-task ``CTABGANMLX``
(``df_ctab_gan_mlx.py``) and the multi-task ``CTABGANMLXMT``
(``df_mt_ctab_gan_mlx.py``):

    1. VGM encoding/decoding for continuous tabular features.
    2. Quality evaluation (diversity / correlation / statistics / validity).
       Ported from the TF multi-task variant with categorical paths
       removed — the MLX variants are continuous-only.
    3. Training-loop callbacks (best-checkpoint tracker, early stopping,
       LR-on-plateau reduction, divergence guard).  Each is a small
       state machine that mutates only what's passed in.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture


# ===========================================================================
# Section 1 — VGM encoding / decoding
# ===========================================================================

# Threshold below which a BGM component is treated as insignificant and
# its mode column is dropped from the encoding.  Matches the existing
# single-task MLX behaviour and keeps the encoded tensor compact.
_BGM_MIN_WEIGHT: float = 1e-3


def fit_vgm_columns(
    df: pd.DataFrame,
    integer_columns: Sequence[str] = (),
    *,
    n_components: int = 10,
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[Dict[str, BayesianGaussianMixture], Dict[str, dict], List[Tuple[int, int]]]:
    """Fit a Bayesian Gaussian Mixture model per continuous column.

    Args:
        df:               Continuous-only DataFrame.  Must already have
                          NaNs filled and dtype coerced.
        integer_columns:  Columns to skip — encoded as a single linear
                          [-1, 1] scalar with no mode head.
        n_components:     BGM component count (max — many will end up
                          weight-zero on real data).
        random_state:     Seed for reproducibility.
        verbose:          Print one line per column fit.

    Returns:
        ``(vgm_models, column_info, continuous_info)`` where:

          * ``vgm_models[col]`` is the fitted ``BayesianGaussianMixture``
            (or ``None`` for skipped/integer columns).
          * ``column_info[col]`` is a dict with min/max/mean/std plus
            ``n_modes`` (the count of significant components used by
            ``transform_vgm``) and ``vgm_components`` (full BGM size).
          * ``continuous_info`` is the per-column ``(scalar_dim, mode_dim)``
            list — ``(1, n_modes)`` for VGM columns, ``(1, 0)`` for
            integer/linear columns — matching the shape the generator
            output heads consume.
    """
    integer_set = set(integer_columns)
    vgm_models: Dict[str, BayesianGaussianMixture] = {}
    column_info: Dict[str, dict] = {}
    continuous_info: List[Tuple[int, int]] = []

    for col in df.columns:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        col_mean = float(df[col].mean())
        col_std = float(df[col].std())

        info: Dict[str, Any] = {
            "type": "continuous",
            "min": col_min,
            "max": col_max,
            "mean": col_mean,
            "std": col_std,
        }

        if col in integer_set:
            if verbose:
                print(f"        VGM {col}... linear (skipped)")
            vgm_models[col] = None
            info["n_modes"] = 0
            info["vgm_components"] = 0
            column_info[col] = info
            continuous_info.append((1, 0))
            continue

        clean = df[col].dropna().values.reshape(-1, 1)
        bgm = BayesianGaussianMixture(
            n_components=n_components,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.001,
            max_iter=100,
            n_init=1,
            random_state=random_state,
        )

        if len(np.unique(clean)) > 1:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                bgm.fit(clean)
            if verbose:
                print(f"        VGM {col}... done")
        else:
            # Constant column — synthesize a single-component fit so the
            # rest of the pipeline can treat it uniformly.
            bgm.means_ = np.array([[clean[0, 0]]])
            bgm.covariances_ = np.array([[[1e-4]]])
            bgm.weights_ = np.array([1.0])
            bgm.n_components = 1
            bgm.predict_proba = lambda x: np.ones((len(x), 1))  # type: ignore[assignment]
            if verbose:
                print(f"        VGM {col}... constant (skipped)")

        n_modes = max(int(np.sum(bgm.weights_ > _BGM_MIN_WEIGHT)), 1)
        vgm_models[col] = bgm
        info["n_modes"] = n_modes
        info["vgm_components"] = int(bgm.n_components)
        column_info[col] = info
        continuous_info.append((1, n_modes))

    return vgm_models, column_info, continuous_info


def transform_vgm(
    df: pd.DataFrame,
    vgm_models: Dict[str, Optional[BayesianGaussianMixture]],
    column_info: Dict[str, dict],
    column_order: Sequence[str],
) -> np.ndarray:
    """Encode a DataFrame to a (rows, total_dim) float32 array.

    For VGM columns the encoding is ``[normalized_scalar, one_hot_mode]``
    where the mode one-hot is sliced to the leading ``n_modes`` columns
    (significant components only).  Integer/linear columns produce a
    single min-max scaled scalar in [-1, 1].
    """
    chunks: List[np.ndarray] = []
    for col in column_order:
        bgm = vgm_models.get(col)
        info = column_info[col]
        values = df[col].values.astype(np.float32)

        if bgm is None or info.get("n_modes", 0) == 0:
            col_min = info["min"]
            col_max = info["max"]
            denom = max(col_max - col_min, 1e-8)
            scalar = (2.0 * (values - col_min) / denom - 1.0).reshape(-1, 1)
            chunks.append(scalar.astype(np.float32))
            continue

        probs = bgm.predict_proba(values.reshape(-1, 1))
        modes = np.argmax(probs, axis=1)
        means = bgm.means_[:, 0]
        stds = np.sqrt(bgm.covariances_[:, 0, 0])
        chosen_means = means[modes]
        chosen_stds = stds[modes]
        normalized = (values - chosen_means) / (4.0 * chosen_stds + 1e-8)
        normalized = np.clip(normalized, -0.99, 0.99).reshape(-1, 1).astype(np.float32)
        one_hot_full = np.eye(bgm.n_components, dtype=np.float32)[modes]
        one_hot_modes = one_hot_full[:, : info["n_modes"]]

        chunks.append(normalized)
        chunks.append(one_hot_modes)

    if not chunks:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.concatenate(chunks, axis=1)


def inverse_transform_vgm(
    generated: np.ndarray,
    vgm_models: Dict[str, Optional[BayesianGaussianMixture]],
    column_info: Dict[str, dict],
    column_order: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Decode a (rows, total_dim) generator output into per-column arrays.

    Output is clipped to each column's training-time ``[min, max]`` so
    downstream consumers (validity checks, scalers) get values inside
    the original range.
    """
    out: Dict[str, np.ndarray] = {}
    offset = 0
    for col in column_order:
        bgm = vgm_models.get(col)
        info = column_info[col]
        n_modes = info.get("n_modes", 0)

        scalar = generated[:, offset]
        offset += 1

        if bgm is None or n_modes == 0:
            col_min = info["min"]
            col_max = info["max"]
            denorm = 0.5 * (scalar + 1.0) * (col_max - col_min) + col_min
        else:
            mode_probs = generated[:, offset : offset + n_modes]
            offset += n_modes
            modes = np.argmax(mode_probs, axis=1)
            means = bgm.means_[:, 0]
            stds = np.sqrt(bgm.covariances_[:, 0, 0])
            denorm = scalar * 4.0 * stds[modes] + means[modes]

        out[col] = np.clip(denorm, info["min"], info["max"]).astype(np.float32)
    return out


# ===========================================================================
# Section 2 — Quality evaluation
# ===========================================================================
#
# Returns the same nested dict shape as the TF multi-task variant so that
# `_compute_overall_score` consumers can be ported without API changes.
# Categorical metric sections collapse to empty dicts since the MLX
# variants are continuous-only.

_DIVERSITY_SAMPLE_CAP: int = 1000
_RNG_SEED: int = 42


def evaluate_with_dataframes(
    real_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    *,
    continuous_columns: Sequence[str],
    column_info: Dict[str, dict],
    column_order: Sequence[str],
) -> Dict[str, Any]:
    """Compute quality metrics comparing real and generated DataFrames.

    Args:
        real_df:             Reference real data.
        generated_df:        Synthetic samples to evaluate.
        continuous_columns:  All columns are continuous in MLX-land —
                             this list drives per-column statistical
                             and validity checks.
        column_info:         Per-column metadata (min/max for validity,
                             stats for normalization).
        column_order:        Canonical column order — both inputs are
                             reindexed to this order before metrics.

    Returns:
        Dict with keys ``diversity``, ``correlation``, ``statistics``,
        ``validity``, ``overall_score``.
    """
    cols = list(column_order)
    real = real_df[cols].copy()
    gen = generated_df[cols].copy()

    metrics: Dict[str, Any] = {}
    metrics["diversity"] = _compute_diversity(real, gen, continuous_columns)
    metrics["correlation"] = _compute_correlation(real, gen, continuous_columns)
    metrics["statistics"] = _compute_statistics(real, gen, continuous_columns)
    metrics["validity"] = _compute_validity(gen, continuous_columns, column_info)
    metrics["overall_score"] = _compute_overall_score(metrics)
    return metrics


def _compute_diversity(
    real: pd.DataFrame,
    gen: pd.DataFrame,
    continuous_columns: Sequence[str],
) -> Dict[str, Any]:
    """Pairwise-distance-based diversity, coverage, and nearest-neighbour."""
    metrics: Dict[str, Any] = {}

    real_arr = real.values.astype(np.float32)
    gen_arr = gen.values.astype(np.float32)

    # Two independent RandomState(42) instances — matches the TF MT
    # implementation byte-for-byte so cross-implementation parity tests
    # pass.  (Yes, this means real and gen get the same permutation when
    # they're the same length, but reproducing TF's exact behaviour is
    # the goal here.)
    n_real = min(len(real_arr), _DIVERSITY_SAMPLE_CAP)
    n_gen = min(len(gen_arr), _DIVERSITY_SAMPLE_CAP)
    if len(real_arr) > n_real:
        real_idx = np.random.RandomState(_RNG_SEED).permutation(len(real_arr))[:n_real]
        real_sample = real_arr[real_idx]
    else:
        real_sample = real_arr
    if len(gen_arr) > n_gen:
        gen_idx = np.random.RandomState(_RNG_SEED).permutation(len(gen_arr))[:n_gen]
        gen_sample = gen_arr[gen_idx]
    else:
        gen_sample = gen_arr

    if len(gen_sample) > 1:
        gen_d = pdist(gen_sample, metric="euclidean")
        metrics["gen_pairwise_distance_mean"] = float(np.mean(gen_d))
        metrics["gen_pairwise_distance_std"] = float(np.std(gen_d))
        metrics["gen_pairwise_distance_min"] = float(np.min(gen_d))
    else:
        metrics["gen_pairwise_distance_mean"] = 0.0
        metrics["gen_pairwise_distance_std"] = 0.0
        metrics["gen_pairwise_distance_min"] = 0.0

    if len(real_sample) > 1:
        real_d = pdist(real_sample, metric="euclidean")
        metrics["real_pairwise_distance_mean"] = float(np.mean(real_d))
    else:
        metrics["real_pairwise_distance_mean"] = 0.0

    if metrics["real_pairwise_distance_mean"] > 0:
        metrics["diversity_ratio"] = (
            metrics["gen_pairwise_distance_mean"]
            / metrics["real_pairwise_distance_mean"]
        )
    else:
        metrics["diversity_ratio"] = 0.0

    coverage_scores: List[float] = []
    for col in continuous_columns:
        if col not in gen.columns:
            continue
        rmn, rmx = float(real[col].min()), float(real[col].max())
        gmn, gmx = float(gen[col].min()), float(gen[col].max())
        rrange = rmx - rmn
        if rrange > 0:
            coverage_scores.append(min(1.0, (gmx - gmn) / rrange))
    metrics["value_space_coverage"] = (
        float(np.mean(coverage_scores)) if coverage_scores else 0.0
    )

    if len(gen_sample) > 0 and len(real_sample) > 0:
        try:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(real_sample)
            distances, _ = nn.kneighbors(gen_sample)
            metrics["nearest_real_distance_mean"] = float(np.mean(distances))
            metrics["nearest_real_distance_std"] = float(np.std(distances))
        except ImportError:
            metrics["nearest_real_distance_mean"] = 0.0
            metrics["nearest_real_distance_std"] = 0.0
    else:
        metrics["nearest_real_distance_mean"] = 0.0
        metrics["nearest_real_distance_std"] = 0.0

    metrics["categorical_uniqueness"] = {}
    return metrics


def _compute_correlation(
    real: pd.DataFrame,
    gen: pd.DataFrame,
    continuous_columns: Sequence[str],
) -> Dict[str, float]:
    """Correlation-matrix preservation between real and generated."""
    cols = [c for c in continuous_columns if c in real.columns]
    if len(cols) < 2:
        return {"correlation_preservation": 1.0, "correlation_error": 0.0}

    real_corr = real[cols].corr().values
    gen_corr = gen[cols].corr().values
    mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
    real_flat = real_corr[mask]
    gen_flat = gen_corr[mask]

    if len(real_flat) == 0:
        return {"correlation_preservation": 1.0, "correlation_error": 0.0}

    real_flat = np.nan_to_num(real_flat, nan=0.0)
    gen_flat = np.nan_to_num(gen_flat, nan=0.0)
    corr_error = float(np.mean(np.abs(real_flat - gen_flat)))

    if np.std(real_flat) > 0 and np.std(gen_flat) > 0:
        corr_corr = np.corrcoef(real_flat, gen_flat)[0, 1]
        preservation = 0.0 if np.isnan(corr_corr) else float(corr_corr)
    else:
        preservation = 0.0

    return {
        "correlation_error": corr_error,
        "correlation_preservation": preservation,
    }


def _compute_statistics(
    real: pd.DataFrame,
    gen: pd.DataFrame,
    continuous_columns: Sequence[str],
) -> Dict[str, Any]:
    """Mean / std error per column, plus overall averages."""
    continuous_stats: Dict[str, dict] = {}
    for col in continuous_columns:
        if col not in real.columns:
            continue
        real_mean = float(real[col].mean())
        real_std = float(real[col].std())
        gen_mean = float(gen[col].mean())
        gen_std = float(gen[col].std())
        rmn = float(real[col].min())
        rmx = float(real[col].max())
        rrange = max(rmx - rmn, abs(real_mean), 1e-8)
        continuous_stats[col] = {
            "mean_error": abs(real_mean - gen_mean) / rrange,
            "std_error": abs(real_std - gen_std) / max(rrange, 1e-8),
            "real_mean": real_mean,
            "gen_mean": gen_mean,
            "real_std": real_std,
            "gen_std": gen_std,
            "real_min": rmn,
            "real_max": rmx,
            "real_range": rrange,
        }

    out: Dict[str, Any] = {"continuous_statistics": continuous_stats}
    if continuous_stats:
        out["mean_error_avg"] = float(
            np.mean([s["mean_error"] for s in continuous_stats.values()])
        )
        out["std_error_avg"] = float(
            np.mean([s["std_error"] for s in continuous_stats.values()])
        )
    else:
        out["mean_error_avg"] = 0.0
        out["std_error_avg"] = 0.0

    out["categorical_statistics"] = {}
    out["categorical_error_avg"] = 0.0
    return out


def _compute_validity(
    gen: pd.DataFrame,
    continuous_columns: Sequence[str],
    column_info: Dict[str, dict],
) -> Dict[str, Any]:
    """Out-of-range counts per continuous column + overall validity score."""
    cont_valid: Dict[str, dict] = {}
    total_invalid = 0
    total_samples = len(gen)

    for col in continuous_columns:
        if col not in gen.columns or col not in column_info:
            continue
        info = column_info[col]
        oor = ((gen[col] < info["min"]) | (gen[col] > info["max"])).sum()
        cont_valid[col] = {
            "out_of_range_count": int(oor),
            "out_of_range_pct": float(oor / total_samples) if total_samples > 0 else 0.0,
        }
        total_invalid += int(oor)

    overall = 1.0 - (total_invalid / total_samples) if total_samples > 0 else 1.0
    return {
        "continuous_validity": cont_valid,
        "categorical_validity": {},
        "overall_validity_score": float(overall),
    }


def _compute_overall_score(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Reduce per-section metrics into the four published scores."""
    diversity_ratio = metrics["diversity"].get("diversity_ratio", 0.0)
    diversity_score = 1.0 - abs(1.0 - diversity_ratio) * 0.5
    diversity_score = max(0.0, min(1.0, diversity_score))

    corr_pres = metrics["correlation"].get("correlation_preservation", 0.0)
    correlation_score = max(0.0, min(1.0, corr_pres))

    mean_err = metrics["statistics"].get("mean_error_avg", 1.0)
    std_err = metrics["statistics"].get("std_error_avg", 1.0)
    cat_err = metrics["statistics"].get("categorical_error_avg", 0.0)
    stat_score = 1.0 / (1.0 + mean_err + std_err + cat_err)
    stat_score = max(0.0, min(1.0, stat_score))

    validity_score = metrics["validity"].get("overall_validity_score", 0.0)

    overall = (
        diversity_score * 0.4
        + correlation_score * 0.4
        + stat_score * 0.15
        + validity_score * 0.05
    )

    return {
        "diversity_score": float(diversity_score),
        "correlation_score": float(correlation_score),
        "statistical_score": float(stat_score),
        "validity_score": float(validity_score),
        "overall_quality": float(overall),
    }


# ===========================================================================
# Section 3 — Training-loop callbacks
# ===========================================================================


class BestCheckpointTracker:
    """Tracks the best score and persists generator/critic weights when it improves.

    Wraps the existing single-task MLX tempdir + safetensors pattern so
    both MLX classes share the same restore behaviour.  Designed for
    MLX modules with ``save_weights`` / ``load_weights`` methods.
    """

    def __init__(
        self,
        *,
        mode: str = "max",
        min_delta: float = 0.0,
        save_dir: Optional[str] = None,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.mode = mode
        self.min_delta = float(min_delta)
        self.save_dir = save_dir or tempfile.mkdtemp(prefix="mlx_ctab_ckpt_")
        os.makedirs(self.save_dir, exist_ok=True)
        self.gen_path = os.path.join(self.save_dir, "best_gen.safetensors")
        self.critic_path = os.path.join(self.save_dir, "best_critic.safetensors")
        self.best: float = float("-inf") if mode == "max" else float("inf")
        self.best_epoch: int = -1
        self.has_checkpoint: bool = False

    def _is_improved(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best + self.min_delta
        return score < self.best - self.min_delta

    def update(self, score: float, epoch: int, gen: Any, critic: Any) -> bool:
        """Save weights if ``score`` improves on ``self.best``.  Returns the improved flag."""
        if not self._is_improved(score):
            return False
        self.best = float(score)
        self.best_epoch = int(epoch)
        try:
            gen.save_weights(self.gen_path)
            critic.save_weights(self.critic_path)
            self.has_checkpoint = True
        except Exception:
            # MLX save_weights occasionally fails on transient I/O — keep
            # ``best`` advanced so future improvements still register, but
            # leave ``has_checkpoint`` false so restore() is a no-op.
            pass
        return True

    def restore(self, gen: Any, critic: Any) -> bool:
        """Reload best weights into the live modules.  Returns True if restored."""
        if not self.has_checkpoint:
            return False
        try:
            gen.load_weights(self.gen_path)
            critic.load_weights(self.critic_path)
            return True
        except Exception:
            return False

    def cleanup(self) -> None:
        shutil.rmtree(self.save_dir, ignore_errors=True)


class EarlyStopping:
    """Counts epochs without improvement; signals stop after ``patience`` flat epochs."""

    def __init__(
        self,
        *,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best: float = float("-inf") if mode == "max" else float("inf")
        self.counter: int = 0

    def update(self, score: float) -> bool:
        """Returns True if patience has been exhausted."""
        if self.mode == "max":
            improved = score > self.best + self.min_delta
        else:
            improved = score < self.best - self.min_delta
        if improved:
            self.best = float(score)
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def soft_reset(self, fraction: float = 0.5) -> None:
        """Halve the counter — used after divergence recovery so the
        recovered model isn't immediately stopped again."""
        self.counter = int(self.counter * fraction)


class ReduceLROnPlateau:
    """Halves optimizer learning rates when a metric plateaus."""

    def __init__(
        self,
        *,
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        mode: str = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = int(patience)
        self.factor = float(factor)
        self.min_lr = float(min_lr)
        self.mode = mode
        self.best: float = float("-inf") if mode == "max" else float("inf")
        self.counter: int = 0

    def update(
        self,
        score: float,
        optimizers: Iterable[Any],
    ) -> Tuple[bool, Optional[float]]:
        """Reduce LR if ``score`` failed to improve for ``patience`` epochs.

        Returns ``(reduced, new_lr)``.  ``optimizers`` must expose a
        mutable ``learning_rate`` attribute (MLX optimizers do).
        """
        if self.mode == "max":
            improved = score > self.best
        else:
            improved = score < self.best
        if improved:
            self.best = float(score)
            self.counter = 0
            return False, None

        self.counter += 1
        if self.counter < self.patience:
            return False, None

        opts = list(optimizers)
        if not opts:
            self.counter = 0
            return False, None

        current_lr = float(opts[0].learning_rate)
        new_lr = max(current_lr * self.factor, self.min_lr)
        if new_lr >= current_lr:
            # Already at the floor — reset the counter so we don't spam
            # log lines every epoch but otherwise no-op.
            self.counter = 0
            return False, None

        for opt in opts:
            opt.learning_rate = new_lr
        self.counter = 0
        return True, new_lr

    def force_reduce(self, optimizers: Iterable[Any]) -> Optional[float]:
        """Unconditionally halve LR — used by divergence recovery."""
        opts = list(optimizers)
        if not opts:
            return None
        current_lr = float(opts[0].learning_rate)
        new_lr = max(current_lr * self.factor, self.min_lr)
        if new_lr >= current_lr:
            return None
        for opt in opts:
            opt.learning_rate = new_lr
        self.counter = 0
        return new_lr


class DivergenceMonitor:
    """Loss-ceiling and degrade-factor guard against runaway training.

    Lifted from the existing single-task MLX inline logic — kept as a
    separate hard stop alongside the eval-quality early stopper because
    catastrophic divergence (NaN losses, huge magnitudes) can happen
    inside an evaluation interval and shouldn't wait for the next eval
    epoch to be detected.
    """

    def __init__(
        self,
        *,
        d_ceil: float = 300.0,
        g_ceil: float = 500.0,
        degrade_factor: float = 10.0,
        patience: int = 5,
    ) -> None:
        self.d_ceil = float(d_ceil)
        self.g_ceil = float(g_ceil)
        self.degrade_factor = float(degrade_factor)
        self.patience = int(patience)
        self.count: int = 0

    def update(self, d_loss: float, g_loss: float, best_score: float) -> bool:
        """Returns True when patience is exhausted (caller should stop)."""
        if not (np.isfinite(d_loss) and np.isfinite(g_loss)):
            self.count += 1
            return self.count >= self.patience

        epoch_score = abs(d_loss) + abs(g_loss)
        is_divergent = (
            (np.isfinite(best_score) and epoch_score > self.degrade_factor * best_score)
            or abs(d_loss) > self.d_ceil
            or abs(g_loss) > self.g_ceil
        )
        if is_divergent:
            self.count += 1
        else:
            self.count = max(0, self.count - 1)
        return self.count >= self.patience

    def reset(self) -> None:
        self.count = 0
