from __future__ import annotations

# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
MLX Regressor factory for NNPredict strategies.

Mirrors NNNClassifierMLX.create_classifier_mlx()/ClassifierTypeMLX so the
selection pattern is consistent across the classifier and regressor families,
and parallels NNPredictRegressor (the Keras factory) on the MLX side.
"""

import sys
from pathlib import Path
from enum import Enum

import mlx.core as mx
import mlx.nn as nn

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from Predictors.MLXRegressorLinear import MLXRegressorLinear


# Dropout matches the Keras NNPredictor's get_estimation_layer (Dropout 0.2).
DROPOUT = 0.2
MIN_FILTER = 32


def nearest_power_of_2(n: int) -> int:
    return 2 ** n.bit_length()


class _LSTMRegressorModel(nn.Module):
    """Hybrid LSTM regressor — same blueprint as the prior classifier work
    (the user reports it performed well there):

        inputs                                       (B, T, num_features)
          → Linear(num_features → num_filters) + tanh   (resize, in parallel)
          ├─ identity (resize_out)
          └─ Conv1d (kernel=2, padding=1, trim to T)
                → Dropout(DROPOUT)
          → Concat([resize_out, conv_out], axis=-1)     (B, T, 2*num_filters)
          → LSTM (returns last timestep)                (B, num_filters)
          → Dropout(DROPOUT)
          → Dense(1, linear)                            (B, 1)

    Resize gives global feature interaction at each timestep; the conv
    branch sees adjacent timesteps for local temporal patterns. Concatenating
    gives the LSTM both views without forcing them through the same channel.
    Different from `add` (the original classifier formulation) — concat
    preserves both signals into separate channels rather than collapsing.
    """

    def __init__(self, seq_len: int, num_features: int, num_filters: int):
        super().__init__()
        self.num_filters = num_filters

        self.resize = nn.Linear(num_features, num_filters)
        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size=2, padding=1)
        self.dropout_conv = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(2 * num_filters, num_filters)
        self.dropout_lstm = nn.Dropout(DROPOUT)
        self.out = nn.Linear(num_filters, 1)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, num_features)
        x_r = mx.tanh(self.resize(x))                  # (B, T, num_filters)
        x_c = self.conv1(x_r)[:, : x_r.shape[1], :]    # (B, T, num_filters)  (trim)
        x_c = self.dropout_conv(x_c)

        concat = mx.concatenate([x_r, x_c], axis=-1)   # (B, T, 2 * num_filters)

        lstm_out, _ = self.lstm(concat)
        last = lstm_out[:, -1, :]                       # (B, num_filters)
        last = self.dropout_lstm(last)

        return self.out(last)


class NNPredictRegressorMLX_LSTM(MLXRegressorLinear):
    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        num_filters = max(MIN_FILTER, nearest_power_of_2(num_features - 1))
        return _LSTMRegressorModel(seq_len, num_features, num_filters)


class RegressorTypeMLX(Enum):
    LSTM = NNPredictRegressorMLX_LSTM


def create_regressor_mlx(
    reg_type: RegressorTypeMLX,
    pair: str,
    nfeatures: int,
    seq_len: int,
    tag: str = "",
):
    """Drop-in mirror of NNNClassifierMLX.create_classifier_mlx()."""
    name = str(reg_type).split(".")[-1]
    reg = reg_type.value(pair, seq_len, nfeatures, tag=tag)
    return reg, name
