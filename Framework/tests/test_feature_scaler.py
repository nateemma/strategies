"""FeatureScaler smoke tests."""

import numpy as np
import pytest

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
