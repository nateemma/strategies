# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
Feature-set selection and per-window scaling/class-weight fitting (research R4, R11).

`fit_scaler`/`class_weights` are the leakage boundary: both take only the caller's
already-role-sliced training rows, never the full frame, so it is structurally hard to
fit on val/test/embargo rows by mistake.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from Framework.FeatureNormalizer import FeatureNormalizer

#: the production include_list (research R11), read from FeatureNormalizer so the study
#: always tracks whatever the live strategies train on.
IND24: List[str] = list(FeatureNormalizer.include_list)

#: research R11 groups, for `ind24-minus:<group>` ablations. This dict is the source of
#: truth for the group *membership*; it is asserted below to partition IND24 exactly.
INDICATOR_GROUPS: dict[str, list[str]] = {
    "trend": ["adx_scaled", "aroonosc_scaled", "ema_fast_norm"],
    "volatility": ["atr_norm", "bb_position", "bb_width"],
    "momentum": [
        "rsi_scaled", "fastk_scaled", "mfi_scaled", "fisher_ss", "cg_ss", "gain_norm",
        "di_diff_scaled",
    ],
    "oscillators": ["cci_scaled", "macd_pos", "macd_neg", "macdhist_norm"],
    "guards": ["guard_metric_pos", "guard_metric_neg"],
    "derived": ["fast_diff", "sar_ratio", "spread_ma", "vwap_pos", "vwap_neg"],
}

_group_union = set()
for _cols in INDICATOR_GROUPS.values():
    _group_union.update(_cols)

if _group_union != set(IND24):
    # Task instruction: if this differs, don't silently patch INDICATOR_GROUPS — surface it.
    raise AssertionError(
        "INDICATOR_GROUPS does not partition FeatureNormalizer.include_list: "
        f"only in groups={_group_union - set(IND24)} only in include_list={set(IND24) - _group_union}"
    )
del _group_union, _cols

#: raw, causal OHLCV-derived columns (research R11); computed by data.py, not here.
RAW8: List[str] = [
    "ret1", "ret4", "log_hc", "log_lc", "log_oc", "range", "logvol_z", "vol_ratio",
]

_PRE_NORMALIZED = set(FeatureNormalizer.pre_normalized_columns)


def columns_for(input_set: str) -> List[str]:
    """Resolve an `input_set` id (data-model.md §ExperimentConfig) to a column list."""
    if input_set == "ind24":
        return list(IND24)
    if input_set == "raw8":
        return list(RAW8)
    if input_set == "both32":
        return list(IND24) + list(RAW8)
    if input_set.startswith("ind24-minus:"):
        group = input_set.split(":", 1)[1]
        if group not in INDICATOR_GROUPS:
            raise ValueError(f"unknown indicator group {group!r} in input_set {input_set!r}")
        drop = set(INDICATOR_GROUPS[group])
        return [c for c in IND24 if c not in drop]
    raise ValueError(f"unrecognised input_set {input_set!r}")


def needs_norm(cols: List[str]) -> List[str]:
    """Columns in `cols` that are not pre-normalized by DataframePopulator.

    `RAW8` columns are never in `FeatureNormalizer.pre_normalized_columns`, so they always
    come out of this filter (research R11: "Scaled with the same per-window RobustScaler").
    """
    return [c for c in cols if c not in _PRE_NORMALIZED]


def fit_scaler(train_df: pd.DataFrame, cols: List[str]) -> RobustScaler:
    """Fit a RobustScaler on `train_df[needs_norm(cols)]` only (train-span rows, R4 #2)."""
    norm_cols = needs_norm(cols)
    scaler = RobustScaler()
    scaler.fit(train_df[norm_cols].to_numpy(dtype=np.float64))
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: RobustScaler, cols: List[str]) -> pd.DataFrame:
    """Return `df[cols]` with the `needs_norm(cols)` subset transformed by `scaler`.

    Transformed columns are clipped to [-10, 10] after `transform`, matching
    FeatureNormalizer.py:398. Passthrough (already pre-normalized) columns are returned
    unchanged. NaNs are filled with 0 after scaling/clipping (production's
    `clean_for_tensor` behaviour) — applied to every returned column, normed or passthrough.
    """
    norm_cols = needs_norm(cols)
    out = df[cols].copy()
    if norm_cols:
        transformed = scaler.transform(out[norm_cols].to_numpy(dtype=np.float64))
        transformed = np.clip(transformed, -10, 10)
        out[norm_cols] = transformed
    return out.fillna(0.0)


def class_weights(train_labels: np.ndarray, n_classes: int = 3) -> List[float]:
    """Inverse-frequency class weights, replicating the single-task branch of
    `Framework/TrainingEngine.py::get_training_class_weights` (lines 85-104):
    `weight[i] = total_samples / (n_classes * count[i])`, 0.0 where a class is absent. This
    is the formula `train()` is given explicitly (research R4 #3) — it is UNNORMALISED,
    unlike the related but distinct `BaseNNStrategy.ratio_to_weights` (lines 753-774), which
    operates on ratios/percentages and divides by the weight sum; that helper is not the one
    used for the per-window weights `train()` receives.
    """
    labels = np.asarray(train_labels)
    if labels.ndim == 2:
        class_indices = np.argmax(labels, axis=1)
    else:
        class_indices = labels.astype(int)

    counts = np.bincount(class_indices, minlength=n_classes)
    total = len(class_indices)

    weights = np.zeros(n_classes)
    for i in range(n_classes):
        if counts[i] > 0:
            weights[i] = total / (n_classes * counts[i])
        else:
            weights[i] = 0.0
    return weights.tolist()
