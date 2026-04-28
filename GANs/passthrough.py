"""
Passthrough columns for GAN augmentation.

GAN generators struggle to reproduce features that are deterministic
functions of the input — calendar features (sin/cos of day-of-year,
day-of-week, minute-of-day), one-hot categoricals, and any feature
with rigid mathematical structure such as ``sin² + cos² = 1``.  The
distribution-matching loss GANs train on doesn't enforce hard
constraints, so the generator approximates them and gets the
structure subtly wrong.  Diagnostics on a real WGAN-GP showed
calendar features systematically biased and their pairwise
correlations attenuated even when other features were fine.

The fix that sidesteps this entirely: don't make the GAN generate
those features at all.  When you build a synthetic batch, replace
the relevant columns with values copied from a real-data sample.
The classifier still sees correct calendar features in synthetic
rows — they're just borrowed from real rows that happened to have
similar conditioning.

This module provides the swap helper used by every augmentation
entry point in the codebase:

  * ``balance_multi_task`` (multi-task WGAN, multi-task CTAB-GAN)
  * ``BaseNNStrategy.wgan_enhance_training_data`` (2D single-task WGAN)
  * ``BaseNNStrategy.wgan_preprocess_training_data`` (2D/3D WGAN)
  * ``BaseNNStrategy.ctab_gan_enhance_training_data`` (CTAB-GAN+)

Usage notes:
  * For 3D ``(N, T, F)`` inputs, the swap copies whole ``(T,)``
    sequences for each passthrough column.  Per-timestep random
    swaps would destroy temporal structure inside the column —
    ``doy_sin`` evolves smoothly across a 16-step window in real
    data, and we want the synthetic version to do the same.
  * For DataFrame inputs, columns can be addressed by name.  For
    ndarray inputs, columns are integer indices.  ``resolve_column_indices``
    converts a list of names to integers given a column-order
    reference.
  * Sampling is uniform with replacement from the real pool.  This
    preserves the marginal distribution of passthrough features but
    doesn't condition on the target class label — that's intentional
    since calendar features are typically near-orthogonal to class
    labels in trading data.  If you need class-conditional sampling,
    the helper is small enough to wrap with a per-class subset filter.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def swap_passthrough_columns(
    synth: Union[np.ndarray, pd.DataFrame],
    real_pool: Union[np.ndarray, pd.DataFrame],
    columns: Sequence[Union[int, str]],
    rng: Optional[np.random.Generator] = None,
) -> Union[np.ndarray, pd.DataFrame]:
    """Replace ``columns`` in ``synth`` with values copied from random
    rows of ``real_pool``.

    Args:
        synth:      Synthetic samples just produced by the GAN.  May be
                    a 2D ndarray ``(n, F)``, 3D ndarray ``(n, T, F)``,
                    or DataFrame ``(n, F)``.  Returned with the same
                    type and shape.
        real_pool:  Real data to draw passthrough values from.  Must
                    have the same trailing-axis feature layout as
                    ``synth``.  For 3D, must also have the same time
                    dimension.
        columns:    Which columns to overwrite.  Integer indices for
                    ndarray inputs, column names (str) for DataFrames.
                    Empty list / None → no-op (returns ``synth``
                    unchanged).
        rng:        Optional ``np.random.Generator``.  Pass a seeded
                    one in tests to make the swap reproducible.

    Returns:
        A copy of ``synth`` with the named columns overwritten.  The
        original ``synth`` is not mutated.

    Raises:
        ValueError: If shapes are incompatible (different feature
                    counts, mismatched time dimensions, etc.).
    """
    if not columns:
        return synth

    if rng is None:
        rng = np.random.default_rng()

    # DataFrame path — column names + .iloc are the natural fit.
    if isinstance(synth, pd.DataFrame):
        if not isinstance(real_pool, pd.DataFrame):
            raise ValueError(
                "swap_passthrough_columns: synth is a DataFrame but real_pool "
                "is not — keep both in the same type for column-name addressing."
            )
        return _swap_dataframe(synth, real_pool, list(columns), rng)

    # ndarray path.
    arr = np.asarray(synth)
    pool = np.asarray(real_pool)
    if isinstance(real_pool, pd.DataFrame):
        # Allow real_pool to be a DataFrame even when synth is an ndarray;
        # this happens during balance_multi_task on tabular backends.
        pool = real_pool.to_numpy()
    if arr.shape[-1] != pool.shape[-1]:
        raise ValueError(
            f"swap_passthrough_columns: feature count mismatch — "
            f"synth has {arr.shape[-1]} features, real_pool has {pool.shape[-1]}"
        )

    col_indices = [int(c) for c in columns]
    return _swap_ndarray(arr, pool, col_indices, rng)


def resolve_column_indices(
    column_names: Sequence[str],
    feature_names: Sequence[str],
) -> List[int]:
    """Translate a list of column names to integer indices using a
    feature-order reference.  Names not found in ``feature_names``
    are silently dropped — caller decides whether to warn.

    The dropped-on-miss behaviour is deliberate: in production the
    config might list calendar columns that are filtered out of the
    feature set in some configurations, and we want the augmentation
    path to keep working rather than crash.  When debugging a missing
    swap, the caller can compare ``len(returned)`` against
    ``len(column_names)``.
    """
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    return [name_to_idx[c] for c in column_names if c in name_to_idx]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _swap_dataframe(
    synth: pd.DataFrame,
    real_pool: pd.DataFrame,
    columns: List[Union[int, str]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """DataFrame-specific implementation.  Resolves int columns to
    names if needed, then assigns by name."""
    # Allow integer column references on DataFrames too — convert
    # them via positional indexing so the caller doesn't have to know
    # the underlying type.
    real_cols = list(real_pool.columns)
    name_columns: List[str] = []
    for c in columns:
        if isinstance(c, int):
            if not (0 <= c < len(real_cols)):
                raise ValueError(
                    f"swap_passthrough_columns: column index {c} out of range "
                    f"for DataFrame with {len(real_cols)} columns"
                )
            name_columns.append(real_cols[c])
        else:
            name_columns.append(str(c))

    missing_real = [c for c in name_columns if c not in real_pool.columns]
    if missing_real:
        raise ValueError(
            f"swap_passthrough_columns: columns {missing_real} not found in real_pool"
        )
    missing_synth = [c for c in name_columns if c not in synth.columns]
    if missing_synth:
        # Silently creating new columns in synth would mask bugs (e.g.
        # the GAN's output schema diverging from the real data) and
        # produce NaN-filled rows after concat.  Surface it loudly.
        raise ValueError(
            f"swap_passthrough_columns: columns {missing_synth} not found "
            f"in synth — generator output schema doesn't match real data"
        )

    n_synth = len(synth)
    n_real = len(real_pool)
    if n_real == 0 or n_synth == 0:
        return synth.copy()

    idx = rng.integers(0, n_real, size=n_synth)
    out = synth.copy()
    sampled = real_pool.iloc[idx][name_columns].reset_index(drop=True)
    sampled.index = out.index  # align so the assignment lands on the right rows
    for col in name_columns:
        out[col] = sampled[col].values
    return out


def _swap_ndarray(
    synth: np.ndarray,
    real_pool: np.ndarray,
    columns: List[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """ndarray-specific implementation handling 2D and 3D shapes."""
    if synth.ndim not in (2, 3):
        raise ValueError(
            f"swap_passthrough_columns: ndim {synth.ndim} not supported "
            f"(must be 2 or 3)"
        )
    if synth.ndim != real_pool.ndim:
        raise ValueError(
            f"swap_passthrough_columns: ndim mismatch — "
            f"synth ndim {synth.ndim}, real_pool ndim {real_pool.ndim}"
        )
    if synth.ndim == 3 and synth.shape[1] != real_pool.shape[1]:
        raise ValueError(
            f"swap_passthrough_columns: time-axis mismatch — "
            f"synth has {synth.shape[1]} timesteps, real_pool has "
            f"{real_pool.shape[1]}"
        )

    n_synth = synth.shape[0]
    n_real = real_pool.shape[0]
    if n_real == 0 or n_synth == 0:
        return synth.copy()

    out = synth.copy()
    idx = rng.integers(0, n_real, size=n_synth)

    if synth.ndim == 2:
        out[:, columns] = real_pool[idx][:, columns]
    else:  # ndim == 3
        # Whole-sequence copy: preserves temporal smoothness within
        # each passthrough column, which matters for features like
        # doy_sin that evolve slowly across a 16-step window.
        out[:, :, columns] = real_pool[idx][:, :, columns]

    return out
