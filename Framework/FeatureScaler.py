"""
FeatureScaler — polymorphic wrapper around an sklearn scaler that handles
both 2D (N, F) dataframes/arrays and 3D (N, T, F) tensors.

For 3D inputs, flattens to (N*T, F), applies the wrapped scaler, and
reshapes back. Same per-feature stats regardless of input shape.

Column-aware: to mirror scale_dataframe/rolling_dataframe_normalise on a
tensor, the wrapped scaler fits + transforms only the "needs_norm" feature
columns and passes "pre_normalized" columns (given by passthrough_indices)
through untouched, then clips to clip_range. With no passthrough_indices it
scales every column (legacy behaviour).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence
import numpy as np
from sklearn.preprocessing import RobustScaler


class FeatureScaler:
    """Polymorphic, column-aware scaler for the post-GAN tensor pipeline.

    The wrapped sklearn scaler's stats fit on per-feature column statistics,
    so the 3D path just needs reshape + apply + reshape back. Picklable
    because the underlying sklearn scaler is picklable.
    """

    def __init__(
        self,
        base: Any | None = None,
        passthrough_indices: Optional[Sequence[int]] = None,
        clip_range: Optional[tuple] = (-10.0, 10.0),
    ) -> None:
        self.base = base if base is not None else RobustScaler()
        self.passthrough_indices = (
            sorted(set(int(i) for i in passthrough_indices))
            if passthrough_indices
            else []
        )
        self.clip_range = clip_range
        # Feature-column indices the base scaler fits/transforms (everything
        # not in passthrough_indices). Resolved at fit against the feature count.
        self._norm_indices: Optional[list] = None

    def _resolve_norm_indices(self, n_features: int) -> list:
        passthrough = set(self.passthrough_indices)
        return [i for i in range(n_features) if i not in passthrough]

    def fit(self, x: np.ndarray) -> "FeatureScaler":
        flat = x.reshape(-1, x.shape[-1]) if x.ndim == 3 else np.asarray(x)
        self._norm_indices = self._resolve_norm_indices(flat.shape[-1])
        if self._norm_indices:
            self.base.fit(flat[:, self._norm_indices])
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        is_3d = x.ndim == 3
        shape = x.shape
        flat = (x.reshape(-1, shape[-1]) if is_3d else np.asarray(x)).astype(
            np.float64, copy=True
        )
        norm_idx = self._norm_indices
        if norm_idx is None:
            # Legacy scaler (fit before column-awareness): scale all columns.
            flat = self.base.transform(flat)
        elif norm_idx:
            flat[:, norm_idx] = self.base.transform(flat[:, norm_idx])
        if self.clip_range is not None:
            flat = np.clip(flat, self.clip_range[0], self.clip_range[1])
        return flat.reshape(shape) if is_3d else flat

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        # Note: the clip in transform() is not invertible; inverse recovers the
        # scaled feature columns only. Not used on the post-GAN critical path.
        is_3d = x.ndim == 3
        shape = x.shape
        flat = (x.reshape(-1, shape[-1]) if is_3d else np.asarray(x)).astype(
            np.float64, copy=True
        )
        norm_idx = self._norm_indices
        if norm_idx is None:
            flat = self.base.inverse_transform(flat)
        elif norm_idx:
            flat[:, norm_idx] = self.base.inverse_transform(flat[:, norm_idx])
        return flat.reshape(shape) if is_3d else flat
