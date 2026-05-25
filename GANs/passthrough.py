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
    feature_names: Optional[Sequence[str]] = None,
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
        columns:    Which columns to overwrite.  May be integer indices,
                    column names (str), or a mix.  Names are resolved to
                    indices in the ndarray path via ``feature_names``,
                    or via ``real_pool.columns`` if ``real_pool`` is a
                    DataFrame.  Empty list / None → no-op (returns
                    ``synth`` unchanged).
        rng:        Optional ``np.random.Generator``.  Pass a seeded
                    one in tests to make the swap reproducible.
        feature_names: Column-order reference used to translate string
                    column names to integer indices in the ndarray
                    path.  Required if ``columns`` contains strings,
                    ``synth`` is an ndarray, AND ``real_pool`` is also
                    an ndarray (i.e. when there's no DataFrame to
                    consult for column order).

    Returns:
        A copy of ``synth`` with the named columns overwritten.  The
        original ``synth`` is not mutated.

    Raises:
        ValueError: If shapes are incompatible (different feature
                    counts, mismatched time dimensions, etc.) or if
                    string columns are passed in the ndarray path
                    without a way to resolve them.
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
    pool_is_df = isinstance(real_pool, pd.DataFrame)
    if pool_is_df:
        # Allow real_pool to be a DataFrame even when synth is an ndarray;
        # this happens during balance_multi_task on tabular backends.
        pool = real_pool.to_numpy()
    else:
        pool = np.asarray(real_pool)
    if arr.shape[-1] != pool.shape[-1]:
        raise ValueError(
            f"swap_passthrough_columns: feature count mismatch — "
            f"synth has {arr.shape[-1]} features, real_pool has {pool.shape[-1]}"
        )

    # Resolve any string column references to indices.  Two sources of
    # column-order truth, in order of preference:
    #   1. real_pool.columns when real_pool is a DataFrame (always correct).
    #   2. feature_names when explicitly passed.
    col_ref: Optional[Sequence[str]] = None
    if pool_is_df:
        col_ref = list(real_pool.columns)
    elif feature_names is not None:
        col_ref = list(feature_names)

    col_indices: List[int] = []
    for c in columns:
        if isinstance(c, str):
            if col_ref is None:
                raise ValueError(
                    f"swap_passthrough_columns: got string column {c!r} but no "
                    "feature_names were provided and real_pool is an ndarray. "
                    "Pass feature_names= so names can be resolved to indices."
                )
            try:
                col_indices.append(col_ref.index(c))
            except ValueError:
                raise ValueError(
                    f"swap_passthrough_columns: column {c!r} not found in "
                    f"feature reference (have {len(col_ref)} columns)"
                ) from None
        else:
            col_indices.append(int(c))

    return _swap_ndarray(arr, pool, col_indices, rng)


def swap_passthrough_columns_nn(
    synth: Union[np.ndarray, pd.DataFrame],
    real_pool: Union[np.ndarray, pd.DataFrame],
    columns: Sequence[Union[int, str]],
    feature_names: Optional[Sequence[str]] = None,
) -> Union[np.ndarray, pd.DataFrame]:
    """Like ``swap_passthrough_columns`` but the synth-to-real pairing is
    nearest-neighbor by non-passthrough feature distance instead of random.

    For each synth row, find the real row whose non-passthrough features
    are closest (euclidean), and copy that real row's passthrough columns
    onto the synth row.  This preserves the joint structure between
    passthrough features and the GAN-generated features — e.g. if a synth
    row has high atr_norm, it gets matched with a real row that also has
    high atr_norm, and inherits that real row's time-of-day fields.

    Only 2D inputs are supported (single-row tabular augmentation).  3D
    sequence inputs fall back to random sampling because per-sequence
    nearest-neighbor on flattened windows is significantly more expensive
    and not needed for the current TAB_DDPM use case.
    """
    if not columns:
        return synth

    # Resolve string column names to integer indices once up front, using
    # the same logic as swap_passthrough_columns so the two functions
    # accept identical inputs.
    if isinstance(synth, pd.DataFrame):
        if not isinstance(real_pool, pd.DataFrame):
            raise ValueError(
                "swap_passthrough_columns_nn: synth is a DataFrame but "
                "real_pool is not."
            )
        col_ref = list(real_pool.columns)
    else:
        arr = np.asarray(synth)
        if arr.ndim == 3:
            # Sequence path — fall back to random sampling for now.
            rng = np.random.default_rng()
            return swap_passthrough_columns(
                synth=synth,
                real_pool=real_pool,
                columns=columns,
                rng=rng,
                feature_names=feature_names,
            )
        if isinstance(real_pool, pd.DataFrame):
            col_ref = list(real_pool.columns)
        elif feature_names is not None:
            col_ref = list(feature_names)
        else:
            col_ref = None

    col_indices: List[int] = []
    for c in columns:
        if isinstance(c, str):
            if col_ref is None:
                raise ValueError(
                    f"swap_passthrough_columns_nn: got string column {c!r} "
                    "but no feature_names were provided."
                )
            try:
                col_indices.append(col_ref.index(c))
            except ValueError:
                raise ValueError(
                    f"swap_passthrough_columns_nn: column {c!r} not found"
                ) from None
        else:
            col_indices.append(int(c))

    # Compute non-passthrough column indices once.
    if isinstance(synth, pd.DataFrame):
        n_features = synth.shape[1]
    else:
        n_features = np.asarray(synth).shape[-1]
    passthrough_set = set(col_indices)
    non_pt = [i for i in range(n_features) if i not in passthrough_set]
    if not non_pt:
        # Edge case — every column is a passthrough, nothing to match on.
        # Fall back to random sampling.
        rng = np.random.default_rng()
        return swap_passthrough_columns(
            synth=synth,
            real_pool=real_pool,
            columns=columns,
            rng=rng,
            feature_names=feature_names,
        )

    # Pull arrays for NN search.
    synth_arr = (
        synth.to_numpy() if isinstance(synth, pd.DataFrame) else np.asarray(synth)
    )
    real_arr = (
        real_pool.to_numpy()
        if isinstance(real_pool, pd.DataFrame)
        else np.asarray(real_pool)
    )

    n_synth = synth_arr.shape[0]
    n_real = real_arr.shape[0]
    if n_real == 0 or n_synth == 0:
        return synth.copy() if hasattr(synth, "copy") else np.array(synth)

    # NN match on the non-passthrough subspace. KDTree (default for low
    # dimensionality) keeps this O(n_synth · log n_real) rather than the
    # O(n_synth · n_real) of a naive scan.
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(real_arr[:, non_pt])
    _, nn_idx = nn.kneighbors(synth_arr[:, non_pt])
    nn_idx = nn_idx[:, 0]

    # Build the output with passthrough columns lifted from matched real rows.
    if isinstance(synth, pd.DataFrame):
        out = synth.copy()
        name_columns = [list(real_pool.columns)[i] for i in col_indices]
        sampled = real_pool.iloc[nn_idx][name_columns].reset_index(drop=True)
        sampled.index = out.index
        for col in name_columns:
            out[col] = sampled[col].values
        return out

    out = synth_arr.copy()
    out[:, col_indices] = real_arr[nn_idx][:, col_indices]
    return out


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
            f"in synth — generator output schema doesn't match real data. "
            f"synth has {len(synth.columns)} columns: {list(synth.columns)}. "
            f"real_pool has {len(real_pool.columns)} columns: {list(real_pool.columns)}"
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
