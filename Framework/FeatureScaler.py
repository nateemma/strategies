"""
FeatureScaler — polymorphic wrapper around an sklearn scaler that handles
both 2D (N, F) dataframes/arrays and 3D (N, T, F) tensors.

For 3D inputs, flattens to (N*T, F), applies the wrapped scaler, and
reshapes back. Same per-feature stats regardless of input shape.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.preprocessing import RobustScaler


class FeatureScaler:
    """Polymorphic scaler for the post-GAN tensor pipeline.

    The wrapped sklearn scaler's stats fit on per-feature column statistics,
    so the 3D path just needs reshape + apply + reshape back. Picklable
    because the underlying sklearn scaler is picklable.
    """

    def __init__(self, base: Any | None = None) -> None:
        self.base = base if base is not None else RobustScaler()

    def fit(self, x: np.ndarray) -> "FeatureScaler":
        flat = x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x
        self.base.fit(flat)
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            out = self.base.transform(flat)
            return out.reshape(shape)
        return self.base.transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            out = self.base.inverse_transform(flat)
            return out.reshape(shape)
        return self.base.inverse_transform(x)
