# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""039 input sets for the plain-LSTM regressors (research R4).

`INPUT_SET` chooses the model's inputs (`include_list`), applied after NNPredictStrategy's own indicators:

  "raw"      raw8 (NNNC/_raw8_features.py: ret1, ret4, log_hc, log_lc, log_oc, range, logvol_z,
             vol_ratio) + NNPredict's current_gain / recent_gain
  "raw_ind"  NNPredict's usual inputs (the shared 24-indicator list + its own gains) + raw8
  "raw_ret"  raw8 + lagged log returns over 8/16/32/64/96 candles + realised volatility over 16/96
             + current_gain / recent_gain

Every added column is normalised causally in its builder (035 R1): a trailing z-score over up to
2880 candles (from 96 on), clipped to +-5 and divided by 5, and registered in `pre_normalized_columns`,
so no fitted scaler ever sees it (constitution XI). Added columns are prefixed (`r8_`, `lr_`) so they
cannot overwrite an existing indicator.

How the inputs are narrowed: the framework requires some indicator columns in `include_list` (its
market-state guards check atr_norm, di_diff_scaled, aroonosc_scaled) and the shared main scaler
transforms its own fitted columns. So `include_list` keeps the parent's columns plus the new ones, and
the model sees only `model_inputs()`: `scale_dataframe` (every place a model tensor is built) and
`get_normalized_size` (the model's input width) select those columns, in that order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from NNNC._raw8_features import RAW8_COLUMNS, add_raw8_features

NORM_WINDOW = 2880
NORM_MIN = 96
LAGS = (8, 16, 32, 64, 96)
VOLS = (16, 96)
GAINS = ["current_gain", "recent_gain"]
R8 = [f"r8_{c}" for c in RAW8_COLUMNS]
LR = [f"lr_ret_{k}" for k in LAGS] + [f"lr_vol_{k}" for k in VOLS]
INPUT_SETS = ("raw", "raw_ind", "raw_ret")


def causal_norm(s: pd.Series) -> pd.Series:
    """Trailing z-score (window up to NORM_WINDOW, from NORM_MIN candles), clipped to +-5, / 5."""
    m = s.rolling(NORM_WINDOW, min_periods=NORM_MIN).mean()
    sd = s.rolling(NORM_WINDOW, min_periods=NORM_MIN).std()
    z = (s - m) / sd.replace(0.0, np.nan)
    return (z.clip(-5.0, 5.0) / 5.0).fillna(0.0).astype(np.float32)


def raw8_columns(df: DataFrame) -> dict[str, pd.Series]:
    r = add_raw8_features(df[["open", "high", "low", "close", "volume"]])
    return {f"r8_{c}": causal_norm(r[c].astype(float)) for c in RAW8_COLUMNS}


def lagged_columns(df: DataFrame) -> dict[str, pd.Series]:
    lc = np.log(df["close"].astype(float))
    r1 = lc.diff()
    out = {f"lr_ret_{k}": causal_norm(lc - lc.shift(k)) for k in LAGS}
    out |= {f"lr_vol_{k}": causal_norm(r1.rolling(k, min_periods=k).std()) for k in VOLS}
    return out


class InputSetMixin:
    INPUT_SET: str = "raw_ind"
    MODEL_KEY_ATTRIBUTES = ("INPUT_SET",)
    # model_inputs shapes the weights, but it is a pure function of INPUT_SET (above) and the family's
    # include_list, so it is classified as covered rather than hashed (hashing re-keys every input-set
    # model, incl. the adopted ones; user decision 2026-10-26). Editing its body = MODEL_CODE_VERSION bump.
    MODEL_KEY_METHODS_IGNORED = ("model_inputs",)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.INPUT_SET not in INPUT_SETS:
            raise ValueError(f"{cls.__name__}: INPUT_SET {cls.INPUT_SET!r} not in {INPUT_SETS}")

    def model_inputs(self) -> list[str]:
        """The columns the model sees, in order."""
        if self.INPUT_SET == "raw":
            return R8 + GAINS
        if self.INPUT_SET == "raw_ret":
            return R8 + LR + GAINS
        return list(self.include_list)  # raw_ind: everything normalised (parent's inputs + raw8)

    def add_additional_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super().add_additional_indicators(dataframe)
        new = raw8_columns(dataframe)
        if self.INPUT_SET == "raw_ret":
            new |= lagged_columns(dataframe)
        for c, v in new.items():
            dataframe[c] = v.to_numpy()
        # instance copies: the shared class-level lists are never mutated
        self.include_list = list(self.include_list) + [c for c in new if c not in self.include_list]
        self.pre_normalized_columns = list(self.pre_normalized_columns) + [
            c for c in new if c not in self.pre_normalized_columns]
        return dataframe

    def scale_dataframe(self, dataframe: DataFrame):
        return super().scale_dataframe(dataframe)[self.model_inputs()]

    def get_normalized_size(self, dataframe: DataFrame):
        return len(self.model_inputs())
