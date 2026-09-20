# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""Tests for `features.fit_scaler`'s `mode` argument (FIX 1: no feature may silently pass
through unscaled).

Synthetic train frame with three columns:
* `normal_col`   -- an ordinary continuous feature, nonzero IQR.
* `zero_iqr_col` -- mostly a constant (gap-fill-bar) value with a handful of real
  outliers, so its 25th/75th-percentile spread (IQR) is exactly 0 but its overall std is
  not -- this mirrors the measured `raw8.log_oc` defect (most bars are open==close).
* `constant_col` -- truly constant over the train rows (IQR and std both 0).

`mode="robust"` must leave every scale exactly as today's `RobustScaler` would (byte-
identical to the running phases); `mode="robust_std"` must substitute `zero_iqr_col`'s std
for its scale and list it in the diagnostic, while `constant_col` stays at scale 1.0 but is
still recorded (separately, as `constant_cols`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.clfstudy import features as F

_COLS = ["normal_col", "zero_iqr_col", "constant_col"]


def _train_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=5.0, scale=2.0, size=n)

    zero_iqr = np.zeros(n)
    zero_iqr[:5] = rng.normal(loc=10.0, scale=1.0, size=5)

    constant = np.full(n, 3.0)

    return pd.DataFrame(
        {"normal_col": normal, "zero_iqr_col": zero_iqr, "constant_col": constant}
    )


def test_default_mode_is_robust():
    df = _train_df()
    default_scaler = F.fit_scaler(df, _COLS)
    explicit_scaler = F.fit_scaler(df, _COLS, mode="robust")
    np.testing.assert_allclose(default_scaler.scale_, explicit_scaler.scale_)
    np.testing.assert_allclose(default_scaler.center_, explicit_scaler.center_)


def test_robust_mode_matches_todays_behaviour_zero_iqr_stays_at_scale_one():
    df = _train_df()
    scaler = F.fit_scaler(df, _COLS, mode="robust")
    idx = _COLS.index("zero_iqr_col")
    # today's (unpatched) RobustScaler behaviour: a zero-IQR column's fitted scale_ is 1.0,
    # i.e. it passes through basically unscaled (the measured defect).
    assert scaler.scale_[idx] == pytest.approx(1.0)


def test_robust_mode_still_reports_the_diagnostic():
    df = _train_df()
    scaler = F.fit_scaler(df, _COLS, mode="robust")
    diag = scaler.clfstudy_diag
    assert diag["mode"] == "robust"
    assert diag["degenerate_iqr_cols"] == ["zero_iqr_col", "constant_col"]
    assert diag["constant_cols"] == ["constant_col"]


def test_robust_std_mode_substitutes_std_for_degenerate_iqr_column():
    df = _train_df()
    scaler = F.fit_scaler(df, _COLS, mode="robust_std")
    idx = _COLS.index("zero_iqr_col")
    expected_std = float(np.std(df["zero_iqr_col"].to_numpy()))
    assert expected_std > 1e-12
    assert scaler.scale_[idx] == pytest.approx(expected_std)


def test_robust_std_mode_constant_column_stays_at_scale_one_but_is_recorded():
    df = _train_df()
    scaler = F.fit_scaler(df, _COLS, mode="robust_std")
    idx = _COLS.index("constant_col")
    assert scaler.scale_[idx] == pytest.approx(1.0)
    assert "constant_col" in scaler.clfstudy_diag["constant_cols"]
    assert "constant_col" in scaler.clfstudy_diag["degenerate_iqr_cols"]


def test_robust_std_mode_diag_shape():
    df = _train_df()
    scaler = F.fit_scaler(df, _COLS, mode="robust_std")
    diag = scaler.clfstudy_diag
    assert diag == {
        "degenerate_iqr_cols": ["zero_iqr_col", "constant_col"],
        "constant_cols": ["constant_col"],
        "mode": "robust_std",
    }


def test_normal_column_identical_scale_under_both_modes():
    df = _train_df()
    s_robust = F.fit_scaler(df, _COLS, mode="robust")
    s_robust_std = F.fit_scaler(df, _COLS, mode="robust_std")
    idx = _COLS.index("normal_col")
    assert s_robust.scale_[idx] == pytest.approx(s_robust_std.scale_[idx])
    assert s_robust.center_[idx] == pytest.approx(s_robust_std.center_[idx])
    assert "normal_col" not in s_robust.clfstudy_diag["degenerate_iqr_cols"]
    assert "normal_col" not in s_robust_std.clfstudy_diag["degenerate_iqr_cols"]


def test_apply_scaler_output_unaffected_by_the_diag_attribute():
    df = _train_df()
    scaler = F.fit_scaler(df, _COLS, mode="robust_std")
    out = F.apply_scaler(df, scaler, _COLS)
    assert list(out.columns) == _COLS
    assert np.isfinite(out.to_numpy()).all()


def test_invalid_mode_rejected():
    df = _train_df()
    with pytest.raises(ValueError):
        F.fit_scaler(df, _COLS, mode="not_a_real_mode")
