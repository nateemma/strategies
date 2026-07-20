"""FeatureScaler smoke tests."""

import numpy as np
import pytest
from sklearn.preprocessing import RobustScaler

from Framework.FeatureScaler import FeatureScaler


def test_2d_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 10, size=(100, 5)).astype(np.float32)
    s = FeatureScaler().fit(x)
    y = s.transform(x)
    z = s.inverse_transform(y)
    np.testing.assert_allclose(x, z, rtol=1e-5)


def test_3d_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 10, size=(50, 16, 5)).astype(np.float32)
    s = FeatureScaler().fit(x)
    y = s.transform(x)
    assert y.shape == x.shape
    z = s.inverse_transform(y)
    np.testing.assert_allclose(x, z, rtol=1e-5)


def test_2d_and_3d_same_stats():
    """Stats fit on 2D vs 3D-flattened-to-2D should match."""
    rng = np.random.default_rng(2)
    x_3d = rng.normal(0, 10, size=(50, 16, 5)).astype(np.float32)
    x_2d_equiv = x_3d.reshape(-1, 5)

    s_3d = FeatureScaler().fit(x_3d)
    s_2d = FeatureScaler().fit(x_2d_equiv)

    np.testing.assert_allclose(s_3d.base.center_, s_2d.base.center_)
    np.testing.assert_allclose(s_3d.base.scale_, s_2d.base.scale_)


def test_cross_shape_transform_consistency():
    """Transforming the same data as 2D vs 3D should give same values (after reshape)."""
    rng = np.random.default_rng(3)
    x_3d = rng.normal(0, 10, size=(50, 16, 5)).astype(np.float32)
    x_2d_equiv = x_3d.reshape(-1, 5)

    s = FeatureScaler().fit(x_3d)
    y_3d = s.transform(x_3d)
    y_2d = s.transform(x_2d_equiv)

    np.testing.assert_allclose(y_3d.reshape(-1, 5), y_2d, rtol=1e-5)


def test_column_aware_passthrough_and_scale():
    """passthrough_indices columns pass through untouched; the rest RobustScale.

    Mirrors scale_dataframe: pre_normalized columns are left in their designed
    range; needs_norm columns get the RobustScaler transform.
    """
    rng = np.random.default_rng(4)
    x = rng.normal(0, 10, size=(200, 5)).astype(np.float32)
    # Column 2 is "pre_normalized" (already in a designed range).
    x[:, 2] = rng.uniform(-1, 1, size=200).astype(np.float32)

    passthrough = [2]
    s = FeatureScaler(passthrough_indices=passthrough).fit(x)
    y = s.transform(x)

    # Passthrough column identical (within clip range, so untouched).
    np.testing.assert_allclose(y[:, 2], x[:, 2], rtol=1e-6)

    # needs_norm columns match a reference RobustScaler fit on just those cols.
    norm_idx = [0, 1, 3, 4]
    ref = RobustScaler().fit(x[:, norm_idx])
    np.testing.assert_allclose(y[:, norm_idx], ref.transform(x[:, norm_idx]), rtol=1e-5)


def test_column_aware_clip():
    """transform clips to ±10 (matches rolling_dataframe_normalise's np.clip)."""
    x = np.zeros((10, 2), dtype=np.float32)
    # Give column 0 an IQR of 1 with a huge outlier so its scaled value >> 10.
    x[:, 0] = np.array([-1, 0, 1, 0, -1, 1, 0, -1, 1, 1e6], dtype=np.float32)
    s = FeatureScaler().fit(x)
    y = s.transform(x)
    assert y.max() <= 10.0 + 1e-6
    assert y.min() >= -10.0 - 1e-6


def test_column_aware_equals_manual_pipeline():
    """Column-aware transform == manual (RobustScale needs_norm + passthrough + clip).

    This is the load-bearing equivalence for path A: normalising a raw tensor
    with this scaler reproduces scale_dataframe per-feature.
    """
    rng = np.random.default_rng(5)
    x = rng.normal(0, 5, size=(300, 6)).astype(np.float32)
    passthrough = [1, 4]
    norm_idx = [0, 2, 3, 5]

    s = FeatureScaler(passthrough_indices=passthrough).fit(x)
    y = s.transform(x)

    manual = x.astype(np.float64).copy()
    ref = RobustScaler().fit(x[:, norm_idx])
    manual[:, norm_idx] = ref.transform(x[:, norm_idx])
    manual = np.clip(manual, -10, 10)

    np.testing.assert_allclose(y, manual, rtol=1e-6)
