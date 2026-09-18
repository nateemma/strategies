from __future__ import annotations

# MLX model variants, mirroring NNNClassifier.py.
#
# Each class inherits MLXClassifierNary and overrides create_model() to
# return an mlx.nn.Module.
#
# Usage (drop-in for NNNClassifier.create_classifier):
#
#   from NNNC.NNNClassifierMLX import create_classifier_mlx, ClassifierTypeMLX
#   clf, name = create_classifier_mlx(ClassifierTypeMLX.LSTM, pair, nfeatures, seq_len, nclasses)
#   clf.set_model_path(path)
#   clf.train(df_train, df_test, train_results, test_results)
#   preds = clf.predict(df)

import sys
from pathlib import Path
from enum import Enum

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from Predictors.MLXClassifierNary import MLXClassifierNary
from Predictors.NoisyCoconut import NoisyCoconutMixin

# -----------------------------------------------------------------------
# Shared constants (mirrors NNNClassifier.py)
# -----------------------------------------------------------------------

DROPOUT = 0.25
MIN_FILTER = 32
L2_STRENGTH = 2e-3  # used conceptually — MLX weight decay is set on the optimizer

# -----------------------------------------------------------------------
# Helper layers
# -----------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """
    Conv1d with causal (left-only) padding.
    Keras: Conv1D(..., padding='causal')
    MLX:   manually pad (kernel_size-1) zeros on the left.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        **kwargs,
    ):
        super().__init__()
        effective_ks = (kernel_size - 1) * dilation + 1
        self.pad_width = effective_ks - 1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            **kwargs,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, seq, channels)
        B, S, C = x.shape
        pad = mx.zeros((B, self.pad_width, C))
        x_padded = mx.concatenate([pad, x], axis=1)
        return self.conv(x_padded)


class TransformerPositionalEmbeddingMLX(nn.Module):
    """Learnable positional-embedding, same concept as TransformerPositionalEmbedding in NNNClassifier."""

    def __init__(self, seq_len: int, embed_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, seq, embed_dim)
        positions = mx.arange(self.seq_len)
        emb = self.pos_emb(positions)  # (seq, embed_dim)
        return x + emb  # broadcast over batch


# spline-based scaling, replacement for final Dense layer
class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size=5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size

        self.base_proj = nn.Linear(input_dim, output_dim)

        # Initialize coefficients: (In, Out, Grid)
        scale = 0.1
        self.spline_coeffs = (
            mx.random.normal((input_dim, output_dim, grid_size)) * scale
        )

    def __call__(self, x):
        # x shape: (Batch, Dim) or (Batch, Seq, Dim)
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.input_dim)  # Flatten to (TotalSamples, InDim)

        # 1. Base Branch: (TotalSamples, OutDim)
        base = self.base_proj(x_flat)

        # 2. Spline Branch
        # x_ext: (TotalSamples, InDim, 1, 1)
        x_ext = x_flat[:, :, None, None]

        # coeffs: (1, InDim, OutDim, Grid)
        coeffs = self.spline_coeffs[None, ...]

        # Calculate basis: (TotalSamples, InDim, OutDim, Grid)
        # We apply the non-linearity to the interaction
        basis = mx.sigmoid(x_ext * coeffs)

        # Sum out the Grid (axis -1) and the InputDim (axis -3)
        # This leaves us with (TotalSamples, OutDim)
        spline_gen = mx.sum(basis, axis=(-1, -3))

        # Combine
        out = base + spline_gen

        # Restore original sequence shape if necessary
        if len(orig_shape) == 3:
            return out.reshape(orig_shape[0], orig_shape[1], self.output_dim)
        return out


# -----------------------------------------------------------------------
# Utility functions shared by models
# -----------------------------------------------------------------------


def nearest_power_of_2(n: int) -> int:
    return 2 ** n.bit_length()


def global_avg_pool_1d(x: mx.array) -> mx.array:
    """Mean over the sequence dimension. (batch, seq, feat) → (batch, feat)"""
    return mx.mean(x, axis=1)


# -----------------------------------------------------------------------
# NNNClassifierMLX_LSTM
# -----------------------------------------------------------------------


class _LSTMModel(nn.Module):
    """
    Mirrors NNNClassifier_LSTM:
      Dense projection → Conv1D → residual → LSTM → Dense bottleneck → softmax
    """

    def __init__(
        self, seq_len: int, num_features: int, num_classes: int, num_filters: int
    ):
        super().__init__()
        self.num_filters = num_filters

        self.resize = nn.Linear(num_features, num_filters)
        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size=2, padding=1)
        self.ln_conv = nn.LayerNorm(num_filters)
        self.dropout_conv = nn.Dropout(DROPOUT)

        self.lstm = nn.LSTM(num_filters, num_filters)
        self.ln_lstm = nn.LayerNorm(num_filters)
        self.dropout_lstm = nn.Dropout(DROPOUT)

        self.bottleneck = nn.Linear(num_filters, num_filters)
        self.dropout_bn = nn.Dropout(DROPOUT)
        self.out = nn.Linear(num_filters, num_classes)

    def encode(self, x: mx.array) -> mx.array:
        # x: (batch, seq, features) → latent (batch, num_filters)
        x_r = nn.relu(self.resize(x))  # (B, S, F)

        # Conv1D block — Conv1d output is (B, S, F) (note: padding=1 for kernel=2 adds 1 extra; trim)
        x_c = self.conv1(x_r)[:, : x_r.shape[1], :]  # keep same seq length
        x_c = self.ln_conv(x_c)
        x_c = self.dropout_conv(x_c)

        # Residual add
        x_res = x_r + x_c  # (B, S, F)

        # LSTM
        lstm_out, _ = self.lstm(x_res)  # (B, S, F)
        lstm_last = lstm_out[:, -1, :]  # (B, F)  — last timestep
        lstm_last = self.ln_lstm(lstm_last)
        lstm_last = self.dropout_lstm(lstm_last)
        return lstm_last

    def decode(self, lstm_last: mx.array) -> mx.array:
        # latent (batch, num_filters) → softmax (batch, num_classes)
        h = nn.elu(self.bottleneck(lstm_last))
        h = self.dropout_bn(h)
        return mx.softmax(self.out(h), axis=-1)

    def __call__(self, x: mx.array) -> mx.array:
        # Split into encode()/decode() so latent-space perturbation
        # (NoisyCoconut) can hook the latent between them. This composition
        # is byte-identical to the original single-pass forward.
        return self.decode(self.encode(x))


class NNNClassifierMLX_LSTM(MLXClassifierNary):
    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        num_filters = max(MIN_FILTER, nearest_power_of_2(num_features - 1))
        return _LSTMModel(seq_len, num_features, self.num_classes, num_filters)


# -----------------------------------------------------------------------
# Architecture study, 2026-09-05. ADDITIVE ONLY — `_LSTMModel` and every
# existing classifier above are untouched, so production behaviour is
# unchanged and these variants are opt-in via `classifier_type`.
#
# Motivation: at the new operating point (0.02 label, guards off, threshold
# 0.80) there is ~25pp of model headroom, where previously there was 0.68pp.
# Architecture has therefore NEVER been testable on realised return in this
# repo — no architecture arm exists in 500+ records. The training curve points
# at UNDERFITTING rather than overfitting: 12,483 parameters, 0.07 per label,
# train loss 0.1568 against val 0.1752 (gap 0.018), plateaued by epoch 30, with
# dropout 0.25 at three points.
#
# Each variant changes exactly ONE thing (Constitution VIII).
# -----------------------------------------------------------------------


class _LSTMModelV2(_LSTMModel):
    """`_LSTMModel` with width, dropout, conv kernel and pooling exposed.

    With the defaults (width_mult=1, dropout=DROPOUT, kernel_size=2,
    pool="last") the forward pass is equivalent to `_LSTMModel` — that
    equivalence is what makes it a valid control for the A/B.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        num_filters: int,
        dropout: float = DROPOUT,
        kernel_size: int = 2,
        pool: str = "last",
        width_mult: int = 1,
        use_conv: bool = True,
        use_resize: bool = True,
        use_residual: bool = True,
    ):
        nf = num_filters * width_mult
        super().__init__(seq_len, num_features, num_classes, nf)
        self.pool = pool
        self.use_conv = use_conv
        self.use_resize = use_resize
        self.use_residual = use_residual

        # Width the conv and the LSTM input operate at. Without the resize the
        # conv runs on the raw feature channels, which keeps the residual add
        # dimensionally valid without reintroducing the projection.
        in_dim = nf if use_resize else num_features
        self.lstm = nn.LSTM(in_dim, nf)
        # padding keeps the sequence length so the residual add stays valid
        self.conv1 = nn.Conv1d(in_dim, in_dim, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.ln_conv = nn.LayerNorm(in_dim)
        self.dropout_conv = nn.Dropout(dropout)
        self.dropout_lstm = nn.Dropout(dropout)
        self.dropout_bn = nn.Dropout(dropout)

    def encode(self, x: mx.array) -> mx.array:
        x_r = nn.relu(self.resize(x)) if self.use_resize else x

        if self.use_conv:
            x_c = self.conv1(x_r)[:, : x_r.shape[1], :]
            x_c = self.ln_conv(x_c)
            x_c = self.dropout_conv(x_c)
            x_res = (x_r + x_c) if self.use_residual else x_c
        else:
            x_res = x_r

        lstm_out, _ = self.lstm(x_res)
        if self.pool == "mean":
            # the last timestep discards 15 of 16 positions the LSTM already
            # computed; mean pooling uses all of them at no extra parameters
            h = mx.mean(lstm_out, axis=1)
        elif self.pool == "max":
            h = mx.max(lstm_out, axis=1)
        else:
            h = lstm_out[:, -1, :]
        h = self.ln_lstm(h)
        return self.dropout_lstm(h)


class _LSTMVariantBase(MLXClassifierNary):
    """Base for the architecture-study variants. One knob each."""

    is_trained = False
    clean_data_required = False

    v_dropout = DROPOUT
    v_kernel = 2
    v_pool = "last"
    v_width = 1
    v_use_conv = True
    v_use_resize = True
    v_use_residual = True

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        num_filters = max(MIN_FILTER, nearest_power_of_2(num_features - 1))
        return _LSTMModelV2(
            seq_len, num_features, self.num_classes, num_filters,
            dropout=self.v_dropout, kernel_size=self.v_kernel,
            pool=self.v_pool, width_mult=self.v_width,
            use_conv=self.v_use_conv, use_resize=self.v_use_resize,
            use_residual=self.v_use_residual,
        )


class NNNClassifierMLX_LSTM_W64(_LSTMVariantBase):
    """CAPACITY: 32 -> 64 filters. Tests the underfitting hypothesis directly."""
    v_width = 2


class NNNClassifierMLX_LSTM_D10(_LSTMVariantBase):
    """REGULARISATION: dropout 0.25 -> 0.10. The 0.018 train/val gap does not
    justify 0.25 at three points."""
    v_dropout = 0.10


class NNNClassifierMLX_LSTM_Pool(_LSTMVariantBase):
    """POOLING: mean over the sequence instead of the last timestep only."""
    v_pool = "mean"


class NNNClassifierMLX_LSTM_NoConv(_LSTMVariantBase):
    """STRAIGHT LSTM: drop the Conv1d + residual block entirely.

    K5 tested a BIGGER conv and lost. This tests the complementary direction --
    whether the conv branch contributes anything at all. Because the block is a
    residual add, the network can already learn to ignore it; if it does, this
    arm should be indistinguishable from the control, and the conv is dead
    weight. If this arm is BETTER, the conv is actively harmful.

    Keeps the resize projection, so the LSTM width is unchanged and this is a
    clean one-thing-at-a-time comparison.
    """
    v_use_conv = False


class NNNClassifierMLX_LSTM_Raw(_LSTMVariantBase):
    """BARE LSTM: raw features straight into the LSTM. No resize, no conv.

    The minimal recurrent baseline -- LSTM(num_features -> num_filters) then the
    same decode head. If this matches the full stack, every pre-LSTM component
    in the production model is decorative.
    """
    v_use_conv = False
    v_use_resize = False


class NNNClassifierMLX_LSTM_NoResize(_LSTMVariantBase):
    """NO RESIZE, CONV KEPT: features -> Conv1d(F->F) -> residual -> LSTM(F->nf).

    The untested cell. `Raw` (no resize, no conv) beat `NoConv` (resize, no conv)
    by 5pp summed, which suggests the resize projection is costing information
    the conv can otherwise use. This isolates that: the conv branch -- shown to
    be load-bearing -- is kept, and only the projection is dropped, with the conv
    running on raw feature channels so the residual add stays valid.
    """
    v_use_resize = False
    v_use_conv = True


class NNNClassifierMLX_LSTM_NoRes(_LSTMVariantBase):
    """NO RESIDUAL: resize -> conv -> LSTM, with the skip connection removed.

    Every conv arm so far kept the residual add, so the block's value could come
    from the conv path, from the skip preserving the unconvolved projection, or
    from their sum. This separates them: if removing the skip is harmless, the
    conv path carries the value; if it is costly, the skip was doing the work
    and the conv is along for the ride.
    """
    v_use_residual = False


class NNNClassifierMLX_LSTM_K5(_LSTMVariantBase):
    """RECEPTIVE FIELD: conv kernel 2 -> 5. At kernel 2 the conv sees two bars
    against a 16-bar sequence, and the residual add lets the network route
    around it entirely."""
    v_kernel = 5


# -----------------------------------------------------------------------
# Stage-2 looped "pondering" (COCONUT) — recurrent latent refinement.
# -----------------------------------------------------------------------


class Ponder(nn.Module):
    """Shared residual refinement block, reused N times on the latent.

    The parameter-tied reuse across steps is the "continuous thought": the model
    refines its latent with the SAME transform before decoding.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def __call__(self, h: mx.array) -> mx.array:
        return h + self.fc2(nn.gelu(self.fc1(self.ln(h))))


class _LSTMPonderModel(_LSTMModel):
    """``_LSTMModel`` + N shared ``Ponder`` steps on the latent before decode.

    ``ponder_steps=0`` skips the loop, so the forward pass is identical to the
    production ``_LSTMModel`` — the matched control for the A/B.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        num_filters: int,
        ponder_steps: int,
    ):
        super().__init__(seq_len, num_features, num_classes, num_filters)
        self.ponder_steps = ponder_steps
        self.ponder = Ponder(num_filters)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.encode(x)
        for _ in range(self.ponder_steps):
            h = self.ponder(h)
        return self.decode(h)


class NNNClassifierMLX_LSTM_Ponder(MLXClassifierNary):
    is_trained = False
    clean_data_required = False
    ponder_steps = 3

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        num_filters = max(MIN_FILTER, nearest_power_of_2(num_features - 1))
        return _LSTMPonderModel(
            seq_len, num_features, self.num_classes, num_filters, self.ponder_steps
        )


# -----------------------------------------------------------------------
# NNNClassifierMLX_LSTM_KAN
# -----------------------------------------------------------------------


class _LSTMKANModel(nn.Module):
    """
    Same as _LSTMModel but replaces the final dense bottleneck/out structure
    with a KANLayer spline-based mapper directly connected to the softmax outputs.
    """

    def __init__(
        self, seq_len: int, num_features: int, num_classes: int, num_filters: int
    ):
        super().__init__()
        self.num_filters = num_filters

        self.resize = nn.Linear(num_features, num_filters)
        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size=2, padding=1)
        self.ln_conv = nn.LayerNorm(num_filters)
        self.dropout_conv = nn.Dropout(DROPOUT)

        self.lstm = nn.LSTM(num_filters, num_filters)
        self.ln_lstm = nn.LayerNorm(num_filters)
        self.dropout_lstm = nn.Dropout(DROPOUT)

        self.kan = KANLayer(num_filters, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x_r = nn.relu(self.resize(x))  # (B, S, F)

        x_c = self.conv1(x_r)[:, : x_r.shape[1], :]
        x_c = self.ln_conv(x_c)
        x_c = self.dropout_conv(x_c)

        x_res = x_r + x_c  # (B, S, F)

        lstm_out, _ = self.lstm(x_res)  # (B, S, F)
        lstm_last = lstm_out[:, -1, :]  # (B, F)
        lstm_last = self.ln_lstm(lstm_last)
        lstm_last = self.dropout_lstm(lstm_last)

        # Output directly through the KAN Layer
        h = self.kan(lstm_last)

        return mx.softmax(h, axis=-1)


class NNNClassifierMLX_LSTM_KAN(MLXClassifierNary):
    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        num_filters = max(MIN_FILTER, nearest_power_of_2(num_features - 1))
        return _LSTMKANModel(seq_len, num_features, self.num_classes, num_filters)


# -----------------------------------------------------------------------
# NNNClassifierMLX_LSTM2  (Two-tier LSTM, mirrors NNNClassifier_LSTM2)
# -----------------------------------------------------------------------


class _LSTM2Model(nn.Module):
    def __init__(
        self, seq_len: int, num_features: int, num_classes: int, num_filters: int
    ):
        super().__init__()
        half = num_filters // 2

        self.resize = nn.Linear(num_features, num_filters)

        self.cnn2 = nn.Conv1d(num_filters, half, kernel_size=2, padding=1)
        self.cnn5 = nn.Conv1d(num_filters, half, kernel_size=5, padding=2)
        self.ln_conv = nn.LayerNorm(num_filters)
        self.drop_conv = nn.Dropout(DROPOUT)

        self.lstm1 = nn.LSTM(num_filters, num_filters)
        self.ln1 = nn.LayerNorm(num_filters)
        self.drop1 = nn.Dropout(DROPOUT)

        self.lstm2 = nn.LSTM(num_filters, num_filters)
        self.ln2 = nn.LayerNorm(num_filters)
        self.drop2 = nn.Dropout(DROPOUT)

        self.bottleneck = nn.Linear(num_filters, num_filters)
        self.drop_bn = nn.Dropout(DROPOUT)
        self.out = nn.Linear(num_filters, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x_r = self.resize(x)  # (B,S,F)

        # Multi-scale CNN — trim to same seq length
        S = x_r.shape[1]
        c2 = nn.relu(self.cnn2(x_r)[:, :S, :])  # (B,S,F//2)
        c5 = nn.relu(self.cnn5(x_r)[:, :S, :])  # (B,S,F//2)
        x_conv = mx.concatenate([c2, c5], axis=-1)  # (B,S,F)
        x_conv = self.ln_conv(x_conv)
        x_conv = self.drop_conv(x_conv)

        x_res = x_r + x_conv  # residual

        # Two-tier LSTM
        out1, _ = self.lstm1(x_res)  # (B,S,F)
        out1 = self.drop1(self.ln1(out1))

        out2, _ = self.lstm2(out1)  # (B,S,F)
        last = out2[:, -1, :]  # (B,F) last timestep
        last = self.drop2(self.ln2(last))

        h = nn.elu(self.bottleneck(last))
        h = self.drop_bn(h)
        return mx.softmax(self.out(h), axis=-1)


class NNNClassifierMLX_LSTM2(MLXClassifierNary):
    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        num_filters = max(2 * MIN_FILTER, nearest_power_of_2(num_features - 1))
        return _LSTM2Model(seq_len, num_features, self.num_classes, num_filters)

    # def calculate_gamma(self, alpha) -> float:
    #     return 2.0  # matches NNNClassifier_LSTM2 override


# -----------------------------------------------------------------------
# NNNClassifierMLX_GRU  (mirrors NNNClassifier_GRU)
# -----------------------------------------------------------------------


class _GRUModel(nn.Module):
    def __init__(
        self, seq_len: int, num_features: int, num_classes: int, num_filters: int
    ):
        super().__init__()
        self.cnn = nn.Conv1d(num_features, num_filters, kernel_size=2, padding=1)
        self.gru = nn.GRU(num_filters, num_filters // 2)

        if seq_len <= 8:
            self.collapse = "avg"
        else:
            self.collapse = "lstm"
            self.lstm_col = nn.LSTM(num_filters // 2, num_filters)

        self.drop = nn.Dropout(DROPOUT)
        self.out = nn.Linear(
            num_filters if seq_len > 8 else num_filters // 2, num_classes
        )

    def __call__(self, x: mx.array) -> mx.array:
        S = x.shape[1]
        x_c = nn.relu(self.cnn(x)[:, :S, :])  # (B,S,F)
        x_g = self.gru(x_c)  # (B,S,F//2) — GRU returns array

        if self.collapse == "avg":
            h = global_avg_pool_1d(x_g)  # (B, F//2)
        else:
            lstm_out, _ = self.lstm_col(x_g)
            h = lstm_out[:, -1, :]  # (B, num_filters)

        h = self.drop(h)
        return mx.softmax(self.out(h), axis=-1)


class NNNClassifierMLX_GRU(MLXClassifierNary):
    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        # GRU hidden = 1024 in Keras version but that's heavy — scale to num_features
        num_filters = min(
            1024, max(MIN_FILTER * 4, nearest_power_of_2(num_features - 1))
        )
        return _GRUModel(seq_len, num_features, self.num_classes, num_filters)


# -----------------------------------------------------------------------
# NNNClassifierMLX_CNN  (mirrors NNNClassifier_CNN)
# -----------------------------------------------------------------------


class _CNNModel(nn.Module):
    def __init__(self, num_features: int, num_classes: int, num_filters: int):
        super().__init__()
        self.conv = CausalConv1d(num_features, num_filters, kernel_size=2)
        self.bn = nn.BatchNorm(num_filters)
        self.drop = nn.Dropout(DROPOUT)
        self.mid = nn.Linear(num_filters, 16)
        self.drop2 = nn.Dropout(DROPOUT)
        self.out = nn.Linear(16, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.tanh(self.conv(x))  # (B, S, num_filters)
        x = self.drop(self.bn(x))
        x = global_avg_pool_1d(x)  # (B, num_filters)
        x = self.mid(x)
        x = self.drop2(x)
        return mx.softmax(self.out(x), axis=-1)


class NNNClassifierMLX_CNN(MLXClassifierNary):
    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        num_filters = max(MIN_FILTER, nearest_power_of_2(num_features - 1))
        return _CNNModel(num_features, self.num_classes, num_filters)


# -----------------------------------------------------------------------
# NNNClassifierMLX_MLP  (mirrors NNNClassifier_MLP)
# -----------------------------------------------------------------------


class _MLPModel(nn.Module):
    def __init__(
        self, seq_len: int, num_features: int, num_classes: int, num_filters: int
    ):
        super().__init__()
        flat_dim = seq_len * num_features
        self.fc1 = nn.Linear(flat_dim, 256)
        self.bn1 = nn.BatchNorm(256)
        self.d1 = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm(128)
        self.d2 = nn.Dropout(DROPOUT)
        self.fc3 = nn.Linear(128, num_filters)
        self.bn3 = nn.BatchNorm(num_filters)
        self.d3 = nn.Dropout(DROPOUT)
        self.out = nn.Linear(num_filters, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        B = x.shape[0]
        x = x.reshape(B, -1)  # flatten
        x = self.d1(self.bn1(nn.relu(self.fc1(x))))
        x = self.d2(self.bn2(nn.relu(self.fc2(x))))
        x = self.d3(self.bn3(nn.relu(self.fc3(x))))
        return mx.softmax(self.out(x), axis=-1)


class NNNClassifierMLX_MLP(MLXClassifierNary):
    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        num_filters = max(MIN_FILTER, nearest_power_of_2(num_features - 1))
        return _MLPModel(seq_len, num_features, self.num_classes, num_filters)


# -----------------------------------------------------------------------
# NNNClassifierMLX_Attention  (mirrors NNNClassifier_Attention)
# -----------------------------------------------------------------------


class _AttentionModel(nn.Module):
    def __init__(
        self, seq_len: int, num_features: int, num_classes: int, layer_size: int
    ):
        super().__init__()
        dropout = 0.2

        self.conv = nn.Conv1d(num_features, layer_size, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm(layer_size)
        self.drop0 = nn.Dropout(dropout)

        # Projections for Q/K/V
        self.q_proj = nn.Linear(layer_size, layer_size)
        self.k_proj = nn.Linear(num_features, layer_size)
        self.v_proj = nn.Linear(num_features, layer_size)

        self.mha = nn.MultiHeadAttention(dims=layer_size, num_heads=8)
        self.drop_a = nn.Dropout(dropout)

        self.ln1 = nn.LayerNorm(layer_size)
        self.ff1 = nn.Linear(layer_size, layer_size)
        self.drop_f = nn.Dropout(dropout)
        self.ff2 = nn.Linear(layer_size, layer_size)

        self.fc = nn.Linear(layer_size, 32)
        self.drop_o = nn.Dropout(0.3)
        self.out = nn.Linear(32, num_classes)

    def __call__(self, x_in: mx.array) -> mx.array:
        # x_in: (B, S, features)
        S = x_in.shape[1]

        x0 = nn.relu(self.conv(x_in)[:, :S, :])  # (B, S, layer_size)
        x0 = self.drop0(self.bn0(x0))

        q = self.q_proj(x0)  # (B, S, layer_size)
        k = self.k_proj(x_in)  # (B, S, layer_size)
        v = self.v_proj(x_in)  # (B, S, layer_size)

        attn = self.mha(q, k, v)  # (B, S, layer_size)
        attn = self.drop_a(attn)
        res = attn + x0  # residual

        # Feed-forward
        x = self.ln1(res)
        x = nn.relu(self.ff1(x))
        x = self.drop_f(x)
        x = self.ff2(x)
        x = x + res

        x = global_avg_pool_1d(x)  # (B, layer_size)
        x = nn.relu(self.fc(x))
        x = self.drop_o(x)
        return mx.softmax(self.out(x), axis=-1)


class NNNClassifierMLX_Attention(MLXClassifierNary):
    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        layer_size = min(nearest_power_of_2(num_features), MIN_FILTER * 2)
        return _AttentionModel(seq_len, num_features, self.num_classes, layer_size)


# -----------------------------------------------------------------------
# NNNClassifierMLX_Multihead  (mirrors NNNClassifier_Multihead)
# -----------------------------------------------------------------------


class _MultiheadModel(nn.Module):
    """Pre-LayerNorm Transformer block with positional embedding."""

    def __init__(
        self, seq_len: int, num_features: int, num_classes: int, layer_size: int
    ):
        super().__init__()
        dropout = 0.3
        num_heads = 4

        self.conv = nn.Conv1d(num_features, layer_size, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm(layer_size)
        self.pos_emb = TransformerPositionalEmbeddingMLX(seq_len, layer_size)

        # Pre-LayerNorm self-attention
        self.ln_attn = nn.LayerNorm(layer_size)
        self.mha = nn.MultiHeadAttention(dims=layer_size, num_heads=num_heads)
        self.drop_a = nn.Dropout(dropout)

        # Pre-LayerNorm FFN
        self.ln_ff = nn.LayerNorm(layer_size)
        self.ff1 = nn.Linear(layer_size, layer_size * 2)
        self.drop_f = nn.Dropout(dropout)
        self.ff2 = nn.Linear(layer_size * 2, layer_size)

        self.fc = nn.Linear(layer_size, 32)
        self.drop_o = nn.Dropout(DROPOUT)
        self.out = nn.Linear(32, num_classes)

    def __call__(self, x_in: mx.array) -> mx.array:
        S = x_in.shape[1]

        x = nn.relu(self.conv(x_in)[:, :S, :])  # (B,S,L)
        x = self.bn0(x)
        x = self.pos_emb(x)  # add positional encoding

        # Self-attention block (pre-LN)
        res = x
        x = self.ln_attn(x)
        x = self.mha(x, x, x)
        x = self.drop_a(x)
        x = x + res

        # FFN block (pre-LN)
        res = x
        x = self.ln_ff(x)
        x = nn.relu(self.ff1(x))
        x = self.drop_f(x)
        x = self.ff2(x)
        x = x + res

        x = global_avg_pool_1d(x)  # (B, L)
        x = nn.relu(self.fc(x))
        x = self.drop_o(x)
        return mx.softmax(self.out(x), axis=-1)


class NNNClassifierMLX_Multihead(MLXClassifierNary):
    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        layer_size = min(nearest_power_of_2(num_features), MIN_FILTER * 2)
        return _MultiheadModel(seq_len, num_features, self.num_classes, layer_size)


# -----------------------------------------------------------------------
# NNNClassifierMLX_Transformer  (mirrors NNNClassifier_Transformer)
# -----------------------------------------------------------------------


class _TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block."""

    def __init__(self, head_size: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()

        self.ln1 = nn.LayerNorm(head_size)
        self.mha = nn.MultiHeadAttention(dims=head_size, num_heads=num_heads)
        self.drop_a = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(head_size)
        self.conv1 = CausalConv1d(head_size, ff_dim, kernel_size=2)
        self.drop_f = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(ff_dim, head_size, kernel_size=2)

    def __call__(self, x: mx.array) -> mx.array:
        res = x
        x = self.ln1(x)
        x = self.mha(x, x, x)
        x = self.drop_a(x)
        x = x + res

        res = x
        x = self.ln2(x)
        x = nn.relu(self.conv1(x))
        x = self.drop_f(x)
        x = self.conv2(x)
        return x + res


class _TransformerModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        head_size: int,
        num_heads: int,
        ff_dim: int,
        num_blocks: int,
        mlp_units: list,
        dropout: float,
        mlp_dropout: float,
    ):
        super().__init__()

        self.proj = nn.Linear(num_features, head_size)
        self.pos_emb = TransformerPositionalEmbeddingMLX(seq_len, head_size)
        self.blocks = [
            _TransformerEncoderBlock(head_size, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ]

        # MLP head
        mlp_layers = []
        in_dim = head_size
        for dim in mlp_units:
            mlp_layers.append(nn.Linear(in_dim, dim))
            in_dim = dim
        self.mlp = mlp_layers
        self.drop_mlp = nn.Dropout(mlp_dropout)
        self.out = nn.Linear(in_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        x = self.pos_emb(x)
        for block in self.blocks:
            x = block(x)
        x = global_avg_pool_1d(x)  # (B, head_size)
        for fc in self.mlp:
            x = self.drop_mlp(fc(x))
        return mx.softmax(self.out(x), axis=-1)


class NNNClassifierMLX_Transformer(MLXClassifierNary):
    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        head_size = max(MIN_FILTER, nearest_power_of_2(num_features - 1))
        num_heads = 4
        ff_dim = 16
        num_blocks = 2
        mlp_units = [head_size, head_size // 2]
        dropout = 0.4
        mlp_dropout = 0.4
        return _TransformerModel(
            seq_len,
            num_features,
            self.num_classes,
            head_size,
            num_heads,
            ff_dim,
            num_blocks,
            mlp_units,
            dropout,
            mlp_dropout,
        )


# -----------------------------------------------------------------------
# NNNClassifierMLX_Wavenet  (mirrors NNNClassifier_Wavenet)
# -----------------------------------------------------------------------


class _WavenetModel(nn.Module):
    def __init__(self, num_features: int, num_classes: int, num_filters: int):
        super().__init__()
        dilations = [1, 2, 4, 8] * 2
        self.convs = [
            CausalConv1d(
                num_features if i == 0 else num_filters,
                num_filters,
                kernel_size=2,
                dilation=r,
            )
            for i, r in enumerate(dilations)
        ]
        self.drop = nn.Dropout(DROPOUT)
        self.out = nn.Linear(num_filters, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        for conv in self.convs:
            x = nn.relu(conv(x))
        x = global_avg_pool_1d(x)
        x = self.drop(x)
        return mx.softmax(self.out(x), axis=-1)


class NNNClassifierMLX_Wavenet(MLXClassifierNary):
    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        num_filters = max(MIN_FILTER, nearest_power_of_2(num_features - 1))
        return _WavenetModel(num_features, self.num_classes, num_filters)


# -----------------------------------------------------------------------
# NNNClassifierMLX_Mamba  (Mamba 1 implementation)
# -----------------------------------------------------------------------
from MLX.mamba_mlx import MambaConfig, Mamba


class _MambaModel(nn.Module):
    def __init__(self, seq_len: int, num_features: int, num_classes: int):
        super().__init__()
        num_filters = max(MIN_FILTER, nearest_power_of_2(num_features - 1))
        self.resize = nn.Linear(num_features, num_filters)
        config = MambaConfig(d_model=num_filters, n_layers=2)
        self.mamba = Mamba(config)
        self.norm = nn.RMSNorm(num_filters)
        self.drop = nn.Dropout(DROPOUT)
        self.kan = KANLayer(num_filters, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x_res = nn.relu(self.resize(x))
        x = self.mamba(x_res)
        x = self.norm(x) + x_res
        x = global_avg_pool_1d(x)
        x = self.drop(x)
        return mx.softmax(self.kan(x), axis=-1)


class NNNClassifierMLX_Mamba(MLXClassifierNary):
    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        return _MambaModel(seq_len, num_features, self.num_classes)


# -----------------------------------------------------------------------
# NNNClassifierMLX_Mamba2  (Mamba 2 implementation)
# -----------------------------------------------------------------------
from MLX.mamba2_mlx import Mamba2Config, Mamba2


class _Mamba2Model(nn.Module):
    def __init__(self, seq_len: int, num_features: int, num_classes: int):
        super().__init__()
        num_filters = max(MIN_FILTER, nearest_power_of_2(num_features - 1))
        self.resize = nn.Linear(num_features, num_filters)
        config = Mamba2Config(
            d_model=num_filters, n_layers=2, d_state=32, headdim=32, ngroups=1
        )
        self.mamba = Mamba2(config)
        self.norm = nn.RMSNorm(num_filters)
        self.drop = nn.Dropout(DROPOUT)
        self.kan = KANLayer(num_filters, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x_res = nn.relu(self.resize(x))
        x = self.mamba(x_res)
        x = self.norm(x) + x_res
        x = global_avg_pool_1d(x)
        x = self.drop(x)
        return mx.softmax(self.kan(x), axis=-1)


class NNNClassifierMLX_Mamba2(MLXClassifierNary):
    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        return _Mamba2Model(seq_len, num_features, self.num_classes)


# -----------------------------------------------------------------------
class _TSMambaModel(nn.Module):
    """
    Bidirectional Time-Series Mamba (TSMamba).
    Learns robust representations from both forward and backward time flows.
    """

    def __init__(self, seq_len: int, num_features: int, num_classes: int):
        super().__init__()
        num_filters = max(MIN_FILTER, nearest_power_of_2(num_features - 1))
        self.resize = nn.Linear(num_features, num_filters)

        # Mamba2 configuration internally loops through multiple layers.
        # For Bidirectional processing, we build two separate unidirectional Mamba blocks.
        config = Mamba2Config(
            d_model=num_filters, n_layers=2, d_state=32, headdim=32, ngroups=1
        )
        self.fwd_mamba = Mamba2(config)
        self.bwd_mamba = Mamba2(config)

        self.norm_fwd = nn.RMSNorm(num_filters)
        self.norm_bwd = nn.RMSNorm(num_filters)
        self.drop = nn.Dropout(DROPOUT)
        self.kan = KANLayer(num_filters, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x_res = nn.relu(self.resize(x))

        # Forward Pass + Norm
        fwd_out = self.norm_fwd(self.fwd_mamba(x_res))

        # Backward Pass + Norm
        x_rev = x_res[:, ::-1, :]
        bwd_out = self.norm_bwd(self.bwd_mamba(x_rev))[:, ::-1, :]

        # Merge via averaging (keeps variance stable) and add residual
        merged = (fwd_out + bwd_out) * 0.5 + x_res

        # Pool to scalar sequence representation
        pooled = global_avg_pool_1d(merged)
        pooled = self.drop(pooled)
        return mx.softmax(self.kan(pooled), axis=-1)


class NNNClassifierMLX_TSMamba(MLXClassifierNary):
    is_trained = False
    clean_data_required = False

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        return _TSMambaModel(seq_len, num_features, self.num_classes)


# -----------------------------------------------------------------------
# NoisyCoconut LSTM variants (training-free — reuse the trained LSTM weights)
# -----------------------------------------------------------------------


class NNNClassifierMLX_LSTM_InJit(NoisyCoconutMixin, NNNClassifierMLX_LSTM):
    """Stage-0 pre-gate: input-space jitter multi-path over the full model."""

    noisy_perturb_space = "input"


class NNNClassifierMLX_LSTM_Noisy(NoisyCoconutMixin, NNNClassifierMLX_LSTM):
    """Stage-1: latent-space (COCONUT) jitter — perturbs the LSTM latent."""

    noisy_perturb_space = "latent"


# -----------------------------------------------------------------------
# NNC classifier study (feature 018) registry types — RIDGE and ROCKET.
# ADDITIVE ONLY: every class above is untouched (contracts/registry.md, research R6).
# -----------------------------------------------------------------------


class _RidgeModel(nn.Module):
    """Flatten(seq_len x num_features) -> Linear(num_classes).

    The L2 penalty ("ridge") is not implemented here — it comes from the shared
    training loop's ``weight_decay`` (``optim.AdamW``); this module is plain
    multinomial logistic regression.
    """

    def __init__(self, seq_len: int, num_features: int, num_classes: int):
        super().__init__()
        self.out = nn.Linear(seq_len * num_features, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        B = x.shape[0]
        x = x.reshape(B, -1)
        return mx.softmax(self.out(x), axis=-1)


class NNNClassifierMLX_Ridge(MLXClassifierNary):
    """RIDGE registry baseline (research R6): L2-penalised multinomial logistic
    regression under the study's shared focal loss / class weights. Takes no `hp`
    keys — the penalty strength is `clf.weight_decay`, set by the harness from
    `ExperimentConfig.train`."""

    is_trained = False
    clean_data_required = False
    hp: dict = {}

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        return _RidgeModel(seq_len, num_features, self.num_classes)


class _RocketModel(nn.Module):
    """MiniRocket-style frozen random-kernel features (research R6).

    ``n_kernels`` kernels of length ``kernel_len``, weights in {-1, 2} (a third of the
    taps at +2, the rest at -1, so each kernel sums to ~0). Each kernel gets a random
    dilation (a power of two, up to ``(seq_len - 1) // (kernel_len - 1)``), a random
    input channel, and a bias drawn ``U(-1, 1)`` — a simplification of MiniRocket's
    quantile-fitted biases, which need a data pass this construction path doesn't have
    (documented in contracts/registry.md). Each kernel's causal (left-zero-padded)
    dilated convolution over its channel is pooled to PPV (fraction of outputs > 0) and
    max over time, giving ``2 * n_kernels`` features for a trainable linear head.

    Kernel weights/channels/bias are ordinary ``mx.array`` attributes (so
    ``save_weights``/``load_weights`` round-trip them across runs) but are frozen with
    ``self.freeze(recurse=False, keys=[...])`` at construction, so
    ``trainable_parameters()`` contains only ``out.weight`` / ``out.bias``.

    ``feature_norm`` ("standardize" | "none", research R6 seed-instability fix):
    the MAX-pooled half of ``feats`` is unbounded and heavy-tailed (measured on real
    LINK/USDT ind24 data: PPV features sit in ``[0, 1]`` with mean ~0.5, but MAX
    features have mean ~1.8 and a tail out to 40-60), unlike PPV which is bounded to
    ``[0, 1]`` by construction. Dotted with the untrained head's random
    ``Linear(2*n_kernels, num_classes)`` weights, that scale/skew makes the *initial*
    (pre-training) softmax collapse to a single class with near-certainty for the large
    majority of rows — e.g. one real-data seed predicted BUY for 99,946 of 99,985 batch
    rows before a single gradient step. Under this codebase's global-norm(1.0) gradient
    clipping that collapse is very hard to train out of within `early_patience`, matching
    the observed ``best_epoch=0`` / val_mcc-never-improves seed failures. When
    ``feature_norm="standardize"`` this normalises ``feats`` (both halves, for
    simplicity/uniformity) to ~zero-mean/unit-std using statistics computed once from the
    first training-mode forward pass (i.e. the first training batch) and then frozen —
    never from val/test data, since normalisation only recomputes while
    ``self.training`` is True and only until calibrated. ``feature_norm="none"``
    reproduces the original (pre-fix) behaviour byte-for-byte, for reproducing historical
    `results.jsonl` rows.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        n_kernels: int = 1000,
        kernel_len: int = 9,
        seed: int | None = None,
        feature_norm: str = "standardize",
    ):
        super().__init__()
        self.kernel_len = kernel_len
        if feature_norm not in ("standardize", "none"):
            raise ValueError(f"feature_norm must be 'standardize' or 'none', got {feature_norm!r}")
        self.feature_norm = feature_norm

        rng = np.random.default_rng(seed)

        n_high = max(1, kernel_len // 3)
        weights = np.full((n_kernels, kernel_len), -1.0, dtype=np.float32)
        for k in range(n_kernels):
            hi = rng.choice(kernel_len, size=n_high, replace=False)
            weights[k, hi] = 2.0

        channels = rng.integers(0, num_features, size=n_kernels).astype(np.int32)

        max_dilation = max(1, (seq_len - 1) // max(1, kernel_len - 1))
        dilation_choices = [1]
        d = 2
        while d <= max_dilation:
            dilation_choices.append(d)
            d *= 2
        dilations = rng.choice(dilation_choices, size=n_kernels).astype(np.int64)

        biases = rng.uniform(-1.0, 1.0, size=n_kernels).astype(np.float32)

        self.kernel_weights = mx.array(weights)
        self.kernel_channels = mx.array(channels)
        self.kernel_bias = mx.array(biases)
        self.freeze(recurse=False, keys=["kernel_weights", "kernel_channels", "kernel_bias"])

        # Plain python/numpy grouping metadata (never an mx.array attribute, so it
        # never enters the module's parameter tree): which kernel indices share a
        # dilation, so the forward pass loops over distinct dilations (typically a
        # handful) instead of once per kernel.
        self._dilation_groups = [
            (int(dval), np.where(dilations == dval)[0].tolist())
            for dval in sorted(set(dilations.tolist()))
        ]

        self.out = nn.Linear(2 * n_kernels, num_classes)

        if self.feature_norm == "standardize":
            n_feats = 2 * n_kernels
            # Non-trainable, save/load-round-tripping buffers (same pattern as the
            # kernel arrays above): identity (mean 0, scale 1) until the first
            # training-mode forward pass calibrates them once, then frozen for the
            # rest of training and for inference.
            self.feat_mean = mx.zeros((n_feats,))
            self.feat_scale = mx.ones((n_feats,))
            self.freeze(recurse=False, keys=["feat_mean", "feat_scale"])
            # Plain python flag (not an mx.array): only gates *when* calibration is
            # (re)computed within this in-memory instance's training run, so it is
            # allowed to reset to False on a fresh load — at that point self.training
            # is False (inference) and the branch below never fires, so the loaded
            # feat_mean/feat_scale are used as-is regardless of this flag's value.
            self._calibrated = False

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, seq, features)
        B, T, _ = x.shape
        ppv_parts = []
        max_parts = []
        for dilation, idx in self._dilation_groups:
            chan = self.kernel_channels[idx]
            w = self.kernel_weights[idx]  # (Kg, kernel_len)
            bias = self.kernel_bias[idx]  # (Kg,)
            x_g = x[:, :, chan]  # (B, T, Kg) — gather each kernel's input channel

            pad = dilation * (self.kernel_len - 1)
            if pad > 0:
                zeros = mx.zeros((B, pad, x_g.shape[-1]))
                x_pad = mx.concatenate([zeros, x_g], axis=1)  # causal left pad
            else:
                x_pad = x_g

            acc = mx.zeros((B, T, x_g.shape[-1]))
            for i in range(self.kernel_len):
                tap = x_pad[:, i * dilation : i * dilation + T, :]
                acc = acc + tap * w[:, i]
            acc = acc - bias

            ppv_parts.append(mx.mean((acc > 0).astype(mx.float32), axis=1))
            max_parts.append(mx.max(acc, axis=1))

        feats = mx.concatenate(ppv_parts + max_parts, axis=-1)  # (B, 2*n_kernels)

        if self.feature_norm == "standardize":
            if self.training and not self._calibrated:
                mean = mx.stop_gradient(mx.mean(feats, axis=0))
                std = mx.stop_gradient(mx.std(feats, axis=0))
                self.feat_mean = mean
                self.feat_scale = mx.maximum(std, mx.array(1e-6))
                self._calibrated = True
            feats = (feats - self.feat_mean) / self.feat_scale

        return mx.softmax(self.out(feats), axis=-1)


class NNNClassifierMLX_Rocket(MLXClassifierNary):
    """ROCKET registry baseline (research R6): frozen random-kernel features + a
    trainable linear head, through the identical shared training loop. `hp` keys:
    `n_kernels` (default 1000), `kernel_len` (default 9), `feature_norm`
    ("standardize" [default, new] | "none" [old, pre-fix behaviour] — see
    `_RocketModel`'s docstring for the seed-instability mechanism this addresses)."""

    is_trained = False
    clean_data_required = False
    hp: dict = {}

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        n_kernels = self.hp.get("n_kernels", 1000)
        kernel_len = self.hp.get("kernel_len", 9)
        feature_norm = self.hp.get("feature_norm", "standardize")
        seed = getattr(self, "train_seed", None)
        return _RocketModel(
            seq_len,
            num_features,
            self.num_classes,
            n_kernels=n_kernels,
            kernel_len=kernel_len,
            seed=seed,
            feature_norm=feature_norm,
        )


# -----------------------------------------------------------------------
# NNNClassifierMLX_LSTM_Fam / GRU_Fam / TCN_Fam / Transformer_Fam / MLP_Fam
# -----------------------------------------------------------------------
#
# Pure single-family registry types (research R7, contracts/registry.md, T034):
# no conv front-end, no residual connections unless `hp["residual"]` says so.
# Every hyperparameter is read from `self.hp` (a dict the harness sets on the
# instance after `create_classifier_mlx`, before `train()`); the defaults below
# are a valid, strategy-usable configuration on their own (FR-015). These are
# additive — no existing class above is modified.


class _LSTMFamModel(nn.Module):
    """Stacked (optionally bidirectional) `nn.LSTM` -> dropout -> Linear -> softmax.

    Pure RNN: no conv front-end, no residual, unlike `_LSTMModel` above.
    `bidirectional=True` runs a second LSTM stack over the time-reversed input
    and concatenates its (re-reversed) per-timestep hidden states with the
    forward stack's, before pooling.

    Two optional `LayerNorm`s bound the two places raw magnitude can otherwise
    drift across a long, unregularized training run (`weight_decay` defaults
    to 0): `input_norm` on the raw per-timestep features before they reach
    `Wx`, and `pool_norm` on the pooled hidden state before the classification
    head -- mirroring `_LSTMModel`'s `ln_conv`/`ln_lstm` pair above (its input
    passes through `resize` + `conv1` + `ln_conv` before the LSTM, and its
    output through `ln_lstm` before the bottleneck). With both enabled
    (the default), `hidden=256` stays finite where it otherwise diverges to
    non-finite predicted probabilities partway through a 40-epoch real-data
    run (both seeds); `hidden=16`/`64` stay finite either way because their
    smaller recurrent weight matrices take far longer to drift that far --
    confirmed via the pre-fix gradient-clip rate, which grows with `hidden`
    (near-zero at 16, ~4-10% of batches at 64, ~10-25% at 256, by epoch
    20-30).

    Both norms are gated by boolean constructor args so they can be disabled
    per hp (research suspicion: `input_norm` applied across the raw 24
    indicator features per bar removes each bar's absolute feature levels --
    e.g. the label-trigger inputs' thresholds -- which is plausibly
    destructive for tabular-like indicator features, and is suspected of
    flattening validation MCC on the harder real-data windows). When
    `input_norm=False` no replacement normalisation is substituted -- the
    features are already RobustScaled and clipped to +-10 upstream of this
    model, so no per-bar, cross-feature renormalisation is needed to keep
    `Wx` inputs bounded; this keeps each bar's absolute feature levels intact.
    When `pool_norm=False` the pooled hidden state is fed to the head as-is.
    Disabling either drops that `LayerNorm`'s parameters entirely (the module
    is omitted, not bypassed), so `hidden=256` runs with both disabled lose
    the divergence guard described above.

    Two further structural options, both default-off, isolate what makes the
    production `_LSTMModel`'s optimisation robust for a GPU ablation:

    `input_proj` (default False): when True, applies `nn.Linear(num_features,
    hidden)` followed by `nn.relu` per time-step, before the recurrent stack
    -- mirroring production's `resize` (`nn.relu(self.resize(x))`), minus the
    Conv1d/residual/LayerNorm that accompany it there. Applied after
    `input_norm` (if enabled) and shared by both directions when
    `bidirectional=True` (the backward branch flips the same projected `x`).
    The recurrent input size becomes `hidden` instead of `num_features`.

    `head` (default "plain"): "plain" is the behaviour above (pooled ->
    optional `pool_norm` -> dropout -> `Linear(out_dim, num_classes)`).
    "bottleneck" mirrors production's `bottleneck`/`out` pair structurally:
    pooled -> optional `pool_norm` -> `Linear(out_dim, bottleneck_dim)` ->
    `nn.elu` (production's bottleneck activation) -> `Dropout(dropout)` ->
    `Linear(bottleneck_dim, num_classes)` -> softmax. Production's own
    `bottleneck` is `nn.Linear(num_filters, num_filters)` (no dimensionality
    reduction), so there is no production sizing rule to mirror exactly;
    `bottleneck_dim = max(8, hidden // 2)` is this module's own choice to give
    "bottleneck" a real bottleneck.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden: int,
        depth: int,
        bidirectional: bool,
        dropout: float,
        pool: str,
        input_norm: bool = False,
        pool_norm: bool = True,
        input_proj: bool = False,
        head: str = "plain",
    ):
        super().__init__()
        if pool not in ("last", "mean"):
            raise ValueError(f"pool must be 'last' or 'mean', got {pool!r}")
        if head not in ("plain", "bottleneck"):
            raise ValueError(f"head must be 'plain' or 'bottleneck', got {head!r}")
        self.pool = pool
        self.bidirectional = bidirectional
        self.use_input_norm = input_norm
        self.use_pool_norm = pool_norm
        self.use_input_proj = input_proj
        self.head_type = head

        if input_norm:
            self.input_norm = nn.LayerNorm(num_features)

        if input_proj:
            self.input_proj = nn.Linear(num_features, hidden)
            rec_in_dim = hidden
        else:
            rec_in_dim = num_features

        in_dim = rec_in_dim
        fwd = []
        for _ in range(depth):
            fwd.append(nn.LSTM(in_dim, hidden))
            in_dim = hidden
        self.fwd_layers = fwd

        if bidirectional:
            in_dim_b = rec_in_dim
            bwd = []
            for _ in range(depth):
                bwd.append(nn.LSTM(in_dim_b, hidden))
                in_dim_b = hidden
            self.bwd_layers = bwd

        out_dim = hidden * (2 if bidirectional else 1)
        if pool_norm:
            self.pool_norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        if head == "bottleneck":
            self.bottleneck_dim = max(8, hidden // 2)
            self.bottleneck = nn.Linear(out_dim, self.bottleneck_dim)
            self.out = nn.Linear(self.bottleneck_dim, num_classes)
        else:
            self.out = nn.Linear(out_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_input_norm:
            x = self.input_norm(x)
        if self.use_input_proj:
            x = nn.relu(self.input_proj(x))

        h = x
        for layer in self.fwd_layers:
            h, _ = layer(h)

        if self.bidirectional:
            hb = mx.flip(x, axis=1)
            for layer in self.bwd_layers:
                hb, _ = layer(hb)
            hb = mx.flip(hb, axis=1)
            h = mx.concatenate([h, hb], axis=-1)

        pooled = mx.mean(h, axis=1) if self.pool == "mean" else h[:, -1, :]
        if self.use_pool_norm:
            pooled = self.pool_norm(pooled)

        if self.head_type == "bottleneck":
            h2 = nn.elu(self.bottleneck(pooled))
            h2 = self.drop(h2)
            return mx.softmax(self.out(h2), axis=-1)

        pooled = self.drop(pooled)
        return mx.softmax(self.out(pooled), axis=-1)


class NNNClassifierMLX_LSTM_Fam(MLXClassifierNary):
    """LSTM_FAM registry type (research R7): pure stacked LSTM, no conv front-end,
    no residual. `hp` keys: `hidden` (64), `depth` (1), `bidirectional` (False),
    `dropout` (0.25), `pool` ('last' | 'mean'), `input_norm` (True), `pool_norm`
    (True) -- see `_LSTMFamModel`'s docstring for what disabling either one does.
    `input_proj` (False), `head` ('plain' | 'bottleneck', default 'plain') --
    default-off structural options for the production-robustness ablation, see
    `_LSTMFamModel`'s docstring."""

    is_trained = False
    clean_data_required = False
    hp: dict = {}

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        hidden = self.hp.get("hidden", 64)
        depth = self.hp.get("depth", 1)
        bidirectional = self.hp.get("bidirectional", False)
        dropout = self.hp.get("dropout", 0.25)
        pool = self.hp.get("pool", "last")
        input_norm = self.hp.get("input_norm", False)
        pool_norm = self.hp.get("pool_norm", True)
        input_proj = self.hp.get("input_proj", False)
        head = self.hp.get("head", "plain")
        return _LSTMFamModel(
            num_features, self.num_classes, hidden, depth, bidirectional, dropout, pool,
            input_norm=input_norm, pool_norm=pool_norm,
            input_proj=input_proj, head=head,
        )


class _GRUFamModel(nn.Module):
    """Identical to `_LSTMFamModel` with `nn.GRU` in place of `nn.LSTM`.

    `nn.GRU.__call__` returns the hidden-state sequence directly (no cell-state
    tuple), unlike `nn.LSTM`. Carries the same optional `input_norm`/`pool_norm`
    `LayerNorm` pair and for the same reason -- see `_LSTMFamModel`'s
    docstring for the full rationale, including what disabling either one does.

    Also carries the same optional `input_proj` (`nn.Linear(num_features,
    hidden)` + `nn.relu` per time-step before the recurrent stack, mirroring
    production's `resize`) and `head` ('plain' | 'bottleneck', where
    'bottleneck' mirrors production's `bottleneck`/`out` pair with
    `bottleneck_dim = max(8, hidden // 2)`) -- see `_LSTMFamModel`'s docstring
    for the full rationale.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden: int,
        depth: int,
        bidirectional: bool,
        dropout: float,
        pool: str,
        input_norm: bool = False,
        pool_norm: bool = True,
        input_proj: bool = False,
        head: str = "plain",
    ):
        super().__init__()
        if pool not in ("last", "mean"):
            raise ValueError(f"pool must be 'last' or 'mean', got {pool!r}")
        if head not in ("plain", "bottleneck"):
            raise ValueError(f"head must be 'plain' or 'bottleneck', got {head!r}")
        self.pool = pool
        self.bidirectional = bidirectional
        self.use_input_norm = input_norm
        self.use_pool_norm = pool_norm
        self.use_input_proj = input_proj
        self.head_type = head

        if input_norm:
            self.input_norm = nn.LayerNorm(num_features)

        if input_proj:
            self.input_proj = nn.Linear(num_features, hidden)
            rec_in_dim = hidden
        else:
            rec_in_dim = num_features

        in_dim = rec_in_dim
        fwd = []
        for _ in range(depth):
            fwd.append(nn.GRU(in_dim, hidden))
            in_dim = hidden
        self.fwd_layers = fwd

        if bidirectional:
            in_dim_b = rec_in_dim
            bwd = []
            for _ in range(depth):
                bwd.append(nn.GRU(in_dim_b, hidden))
                in_dim_b = hidden
            self.bwd_layers = bwd

        out_dim = hidden * (2 if bidirectional else 1)
        if pool_norm:
            self.pool_norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        if head == "bottleneck":
            self.bottleneck_dim = max(8, hidden // 2)
            self.bottleneck = nn.Linear(out_dim, self.bottleneck_dim)
            self.out = nn.Linear(self.bottleneck_dim, num_classes)
        else:
            self.out = nn.Linear(out_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_input_norm:
            x = self.input_norm(x)
        if self.use_input_proj:
            x = nn.relu(self.input_proj(x))

        h = x
        for layer in self.fwd_layers:
            h = layer(h)

        if self.bidirectional:
            hb = mx.flip(x, axis=1)
            for layer in self.bwd_layers:
                hb = layer(hb)
            hb = mx.flip(hb, axis=1)
            h = mx.concatenate([h, hb], axis=-1)

        pooled = mx.mean(h, axis=1) if self.pool == "mean" else h[:, -1, :]
        if self.use_pool_norm:
            pooled = self.pool_norm(pooled)

        if self.head_type == "bottleneck":
            h2 = nn.elu(self.bottleneck(pooled))
            h2 = self.drop(h2)
            return mx.softmax(self.out(h2), axis=-1)

        pooled = self.drop(pooled)
        return mx.softmax(self.out(pooled), axis=-1)


class NNNClassifierMLX_GRU_Fam(MLXClassifierNary):
    """GRU_FAM registry type (research R7): pure stacked GRU, no conv front-end,
    no residual. Same `hp` keys as LSTM_FAM: `hidden` (64), `depth` (1),
    `bidirectional` (False), `dropout` (0.25), `pool` ('last' | 'mean'),
    `input_norm` (True), `pool_norm` (True) -- see `_LSTMFamModel`'s docstring
    for what disabling either one does. `input_proj` (False), `head`
    ('plain' | 'bottleneck', default 'plain') -- default-off structural
    options for the production-robustness ablation, see `_LSTMFamModel`'s
    docstring."""

    is_trained = False
    clean_data_required = False
    hp: dict = {}

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        hidden = self.hp.get("hidden", 64)
        depth = self.hp.get("depth", 1)
        bidirectional = self.hp.get("bidirectional", False)
        dropout = self.hp.get("dropout", 0.25)
        pool = self.hp.get("pool", "last")
        input_norm = self.hp.get("input_norm", False)
        pool_norm = self.hp.get("pool_norm", True)
        input_proj = self.hp.get("input_proj", False)
        head = self.hp.get("head", "plain")
        return _GRUFamModel(
            num_features, self.num_classes, hidden, depth, bidirectional, dropout, pool,
            input_norm=input_norm, pool_norm=pool_norm,
            input_proj=input_proj, head=head,
        )


class _TCNBlock(nn.Module):
    """One causal dilated-conv layer of the TCN stack.

    Activation is ReLU, matching this file's existing causal-conv models
    (`_WavenetModel`, `_CNNModel`'s conv path uses tanh but the wavenet/dilated
    stack uses ReLU — that convention is followed here). `residual=True` adds a
    1x1-projected skip connection (`nn.Linear`, since the conv operates on the
    channel-last `(B, S, C)` layout) when the block changes channel count.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        dilation: int,
        dropout: float,
        residual: bool,
    ):
        super().__init__()
        self.residual = residual
        self.conv = CausalConv1d(
            in_channels, out_channels, kernel_size=kernel, dilation=dilation
        )
        self.drop = nn.Dropout(dropout)
        if residual:
            self.proj = (
                nn.Linear(in_channels, out_channels)
                if in_channels != out_channels
                else None
            )

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.relu(self.conv(x))
        h = self.drop(h)
        if self.residual:
            skip = self.proj(x) if self.proj is not None else x
            h = h + skip
        return h


class _TCNFamModel(nn.Module):
    """Causal dilated Conv1d stack -> pool over time -> Linear -> softmax.

    Layer `i`'s dilation is `dilation_base ** i` (constant 1 at the first layer
    regardless of base, and constant 1 throughout when `dilation_base=1`).

    `dilations` (list[int], default None) overrides `depth`/`dilation_base`
    entirely when given: the stack has `len(dilations)` blocks, using these
    dilations in order (e.g. Wavenet's `[1, 2, 4, 8, 1, 2, 4, 8]`). Must be a
    non-empty list of positive ints. When `None`, the `depth`/`dilation_base`
    behaviour above is unchanged.

    `pool="mean"` (default) global-average-pools over time. `pool="last"`
    instead takes the final time-step's vector — the window's most recent
    bar, where the label trigger sits — before the head. Pooling adds no
    parameters either way.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        channels: int,
        kernel: int,
        depth: int,
        dilation_base: int,
        residual: bool,
        dropout: float,
        pool: str = "mean",
        dilations: list | None = None,
    ):
        super().__init__()
        if pool not in ("last", "mean"):
            raise ValueError(f"pool must be 'last' or 'mean', got {pool!r}")
        self.pool = pool
        if dilations is not None:
            if not dilations or any(not isinstance(d, int) or d <= 0 for d in dilations):
                raise ValueError(
                    f"dilations must be a non-empty list of positive ints, got {dilations!r}"
                )
            layer_dilations = list(dilations)
        else:
            layer_dilations = [dilation_base**i for i in range(depth)]
        blocks = []
        in_ch = num_features
        for dilation in layer_dilations:
            blocks.append(_TCNBlock(in_ch, channels, kernel, dilation, dropout, residual))
            in_ch = channels
        self.blocks = blocks
        self.out = nn.Linear(channels, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for block in self.blocks:
            h = block(h)
        h = global_avg_pool_1d(h) if self.pool == "mean" else h[:, -1, :]
        return mx.softmax(self.out(h), axis=-1)


class NNNClassifierMLX_TCN_Fam(MLXClassifierNary):
    """TCN_FAM registry type (research R7): causal dilated-conv stack, pooled
    head, no LSTM/attention. `hp` keys: `channels` (64), `kernel` (3), `depth` (3),
    `dilation_base` (1), `residual` (False), `dropout` (0.25), `pool`
    ('mean' (default) | 'last'), `dilations` (list[int], default None — when
    given, overrides `depth`/`dilation_base`: the stack has `len(dilations)`
    blocks with these dilations in order, e.g. Wavenet's
    `[1, 2, 4, 8, 1, 2, 4, 8]`; must be a non-empty list of positive ints).

    Wavenet equivalence: `{"channels": 32, "kernel": 2, "dilations":
    [1, 2, 4, 8, 1, 2, 4, 8]}` (32 = `_WavenetModel`'s channel count for 24
    features, via the same `max(MIN_FILTER, nearest_power_of_2(num_features -
    1))` rule) reproduces `_WavenetModel`'s dilated-conv stack as closely as
    this family's code allows: same `CausalConv1d` padding, same per-layer
    ReLU activation placement, same `channels`/`kernel`/dilation schedule, and
    (at `pool='mean'`, the default) the same pool-then-`Linear` head.
    Remaining differences from `_WavenetModel`: (1) dropout is applied once
    per block here (`_TCNBlock`, `dropout=0.25` default) vs. Wavenet, which has
    no per-layer dropout and instead drops once after pooling; (2) `residual`
    defaults to `False` here (matching Wavenet, which has no skip
    connections), but this family additionally supports `residual=True`,
    which Wavenet does not have."""

    is_trained = False
    clean_data_required = False
    hp: dict = {}

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        channels = self.hp.get("channels", 64)
        kernel = self.hp.get("kernel", 3)
        depth = self.hp.get("depth", 3)
        dilation_base = self.hp.get("dilation_base", 1)
        residual = self.hp.get("residual", False)
        dropout = self.hp.get("dropout", 0.25)
        pool = self.hp.get("pool", "mean")
        dilations = self.hp.get("dilations", None)
        return _TCNFamModel(
            num_features,
            self.num_classes,
            channels,
            kernel,
            depth,
            dilation_base,
            residual,
            dropout,
            pool,
            dilations=dilations,
        )


class _RoPEMultiHeadAttention(nn.Module):
    """Self-attention with rotary position embeddings applied to the per-head
    query/key projections.

    `nn.MultiHeadAttention` hides its internal q/k projections behind a single
    call, so RoPE cannot be applied to them there (see `_TransformerFamBlock`,
    which used to apply `nn.RoPE` once to the shared `d_model` sequence before
    the first block — not a per-head rotary embedding at all). This class owns
    separate `Linear(d_model, d_model)` projections for q/k/v and the output,
    reshapes each to `(B, heads, T, head_dim)`, rotates q and k with
    `nn.RoPE(head_dim, traditional=False)` (same position ids per head, offset
    always 0 — every window is encoded from scratch, matching the model's
    fixed-length sliding-window usage), then runs scaled dot-product attention
    via `mx.fast.scaled_dot_product_attention` with no mask — consistent with
    `_TransformerFamBlock`'s existing (unmasked) `nn.MultiHeadAttention` call,
    since this is classification over the whole window, not autoregressive
    generation.
    """

    def __init__(self, d_model: int, heads: int):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = d_model // heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = nn.RoPE(self.head_dim, traditional=False)

    def _split_heads(self, x: mx.array, T: int) -> mx.array:
        B = x.shape[0]
        return x.reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, queries: mx.array, keys: mx.array, values: mx.array) -> mx.array:
        Tq, Tk = queries.shape[1], keys.shape[1]
        q = self._split_heads(self.q_proj(queries), Tq)
        k = self._split_heads(self.k_proj(keys), Tk)
        v = self._split_heads(self.v_proj(values), Tk)
        q = self.rope(q)
        k = self.rope(k)
        scale = self.head_dim**-0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(queries.shape[0], Tq, self.heads * self.head_dim)
        return self.out_proj(out)


class _TransformerFamBlock(nn.Module):
    """Pre-LN transformer encoder block: LN -> attn -> residual, LN -> FFN(GELU) -> residual.

    A plain `Linear` FFN (unlike `_TransformerEncoderBlock`'s causal-conv FFN
    above) — kept as a separate class rather than reusing that block so its
    FFN/dropout wiring matches the TRANSFORMER_FAM contract exactly.

    Takes a pre-built `attn` module (`nn.MultiHeadAttention` or
    `_RoPEMultiHeadAttention`) rather than building one itself, so
    `_TransformerFamModel` can choose the attention implementation per `pos`.
    """

    def __init__(self, attn: nn.Module, d_model: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = attn
        self.drop_a = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model * ffn_mult)
        self.ff2 = nn.Linear(d_model * ffn_mult, d_model)
        self.drop_f = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        res = x
        h = self.ln1(x)
        h = self.mha(h, h, h)
        h = self.drop_a(h)
        x = res + h

        res = x
        h = self.ln2(x)
        h = self.ff2(nn.gelu(self.ff1(h)))
        h = self.drop_f(h)
        return res + h


class _TransformerFamModel(nn.Module):
    """Linear(F->d_model) -> positional encoding (`pos`) -> `layers` pre-LN blocks ->
    pool over time -> Linear -> softmax.

    `pos="rope"` applies true rotary position embedding: each block's attention
    is a `_RoPEMultiHeadAttention`, which rotates its own per-head q/k
    projections every block (not a single shared rotation of the input
    sequence applied once up front, as an earlier version of this model did).
    `pos="learned"` instead adds a learned positional embedding once before the
    first block, and every block then uses plain `nn.MultiHeadAttention`.
    `pos="none"` uses no positional signal at all.

    `pool="mean"` (default) global-average-pools over time. `pool="last"`
    instead takes the final time-step's vector — the window's most recent
    bar, where the label trigger sits — before the head. Pooling adds no
    parameters either way.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        d_model: int,
        heads: int,
        layers: int,
        ffn_mult: int,
        pos: str,
        dropout: float,
        pool: str = "mean",
    ):
        super().__init__()
        if pos not in ("learned", "none", "rope"):
            raise ValueError(f"pos must be 'learned', 'none' or 'rope', got {pos!r}")
        if pool not in ("last", "mean"):
            raise ValueError(f"pool must be 'last' or 'mean', got {pool!r}")
        self.pos = pos
        self.pool = pool
        self.proj = nn.Linear(num_features, d_model)
        if pos == "learned":
            self.pos_emb = TransformerPositionalEmbeddingMLX(seq_len, d_model)
        if pos == "rope":
            self.blocks = [
                _TransformerFamBlock(
                    _RoPEMultiHeadAttention(d_model, heads), d_model, ffn_mult, dropout
                )
                for _ in range(layers)
            ]
        else:
            self.blocks = [
                _TransformerFamBlock(
                    nn.MultiHeadAttention(dims=d_model, num_heads=heads),
                    d_model,
                    ffn_mult,
                    dropout,
                )
                for _ in range(layers)
            ]
        self.out = nn.Linear(d_model, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.proj(x)
        if self.pos == "learned":
            h = self.pos_emb(h)
        for block in self.blocks:
            h = block(h)
        h = global_avg_pool_1d(h) if self.pool == "mean" else h[:, -1, :]
        return mx.softmax(self.out(h), axis=-1)


class NNNClassifierMLX_Transformer_Fam(MLXClassifierNary):
    """TRANSFORMER_FAM registry type (research R7): pre-LN encoder, pooled over
    time, no conv front-end. `hp` keys: `d_model` (64), `heads` (4), `layers` (2),
    `ffn_mult` (2), `pos` ('learned' | 'none' | 'rope'), `dropout` (0.25), `pool`
    ('mean' (default) | 'last')."""

    is_trained = False
    clean_data_required = False
    hp: dict = {}

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        d_model = self.hp.get("d_model", 64)
        heads = self.hp.get("heads", 4)
        layers = self.hp.get("layers", 2)
        ffn_mult = self.hp.get("ffn_mult", 2)
        pos = self.hp.get("pos", "learned")
        dropout = self.hp.get("dropout", 0.25)
        pool = self.hp.get("pool", "mean")
        return _TransformerFamModel(
            seq_len,
            num_features,
            self.num_classes,
            d_model,
            heads,
            layers,
            ffn_mult,
            pos,
            dropout,
            pool,
        )


class _MLPFamModel(nn.Module):
    """Flatten -> `depth` x (Linear(width) -> ReLU -> Dropout) -> Linear -> softmax.

    Pure MLP: no BatchNorm (unlike `_MLPModel` above), no residual.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        width: int,
        depth: int,
        dropout: float,
    ):
        super().__init__()
        flat_dim = seq_len * num_features
        layers = []
        in_dim = flat_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            in_dim = width
        self.layers = layers
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(in_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        B = x.shape[0]
        h = x.reshape(B, -1)
        for layer in self.layers:
            h = self.drop(nn.relu(layer(h)))
        return mx.softmax(self.out(h), axis=-1)


class NNNClassifierMLX_MLP_Fam(MLXClassifierNary):
    """MLP_FAM registry type (research R7): pure Flatten -> MLP -> softmax, no
    BatchNorm/residual. `hp` keys: `width` (256), `depth` (2), `dropout` (0.25)."""

    is_trained = False
    clean_data_required = False
    hp: dict = {}

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        width = self.hp.get("width", 256)
        depth = self.hp.get("depth", 2)
        dropout = self.hp.get("dropout", 0.25)
        return _MLPFamModel(seq_len, num_features, self.num_classes, width, depth, dropout)


# -----------------------------------------------------------------------
# NNNClassifierMLX_TSMixer  (TSMIXER registry type, research R13 / Part B ref [8])
# -----------------------------------------------------------------------


class _TSMixerBlock(nn.Module):
    """One TSMixer block: time-mixing MLP across the time axis, then
    feature-mixing MLP across the feature axis, each pre-LN + residual.

    Time-mixing: LayerNorm(F) -> transpose (B,T,F)->(B,F,T) -> Linear(T,T) ->
    GELU -> dropout -> transpose back -> residual.
    Feature-mixing: LayerNorm(F) -> Linear(F, F*ff_mult) -> GELU -> dropout ->
    Linear(F*ff_mult, F) -> residual.
    """

    def __init__(self, seq_len: int, num_features: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln_time = nn.LayerNorm(num_features)
        self.time_lin = nn.Linear(seq_len, seq_len)
        self.drop_time = nn.Dropout(dropout)
        self.ln_feat = nn.LayerNorm(num_features)
        self.feat_lin1 = nn.Linear(num_features, num_features * ff_mult)
        self.feat_lin2 = nn.Linear(num_features * ff_mult, num_features)
        self.drop_feat = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        res = x
        h = self.ln_time(x)
        h = h.transpose(0, 2, 1)  # (B, F, T)
        h = nn.gelu(self.time_lin(h))
        h = self.drop_time(h)
        h = h.transpose(0, 2, 1)  # (B, T, F)
        x = res + h

        res = x
        h = self.ln_feat(x)
        h = nn.gelu(self.feat_lin1(h))
        h = self.drop_feat(h)
        h = self.feat_lin2(h)
        return res + h


class _TSMixerModel(nn.Module):
    """`blocks` `_TSMixerBlock`s, then mean-pool over time -> Linear -> softmax.

    Read-out: mean-pool over time (not flatten), matching the
    `global_avg_pool_1d` convention every other _Fam model in this file uses --
    keeps the head's parameter count independent of `seq_len`.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        blocks: int,
        ff_mult: int,
        dropout: float,
    ):
        super().__init__()
        self.blocks = [
            _TSMixerBlock(seq_len, num_features, ff_mult, dropout) for _ in range(blocks)
        ]
        self.out = nn.Linear(num_features, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x)
        h = global_avg_pool_1d(x)
        return mx.softmax(self.out(h), axis=-1)


class NNNClassifierMLX_TSMixer(MLXClassifierNary):
    """TSMIXER registry type (research R13 / Part B ref [8]): interleaved
    time-mixing / feature-mixing MLP blocks (TSMixer). `hp` keys: `blocks` (2),
    `ff_mult` (2), `dropout` (0.25)."""

    is_trained = False
    clean_data_required = False
    hp: dict = {}

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        blocks = self.hp.get("blocks", 2)
        ff_mult = self.hp.get("ff_mult", 2)
        dropout = self.hp.get("dropout", 0.25)
        return _TSMixerModel(
            seq_len, num_features, self.num_classes, blocks, ff_mult, dropout
        )


# -----------------------------------------------------------------------
# NNNClassifierMLX_ITransformer  (ITRANSFORMER registry type, research R13 /
# Part B ref [7])
# -----------------------------------------------------------------------


class _ITransformerAttention(nn.Module):
    """Explicit (non-hidden) multi-head self-attention over the *feature* axis.

    `nn.MultiHeadAttention` hides its attention weights, and iTransformer needs
    them for the secondary importance signal (`model.last_attention`), so this
    mirrors `_RoPEMultiHeadAttention`'s q/k/v-projection pattern but without any
    positional encoding (the tokens here are unordered features, not time
    steps) and exposes the softmax weights via `last_attn`.
    """

    def __init__(self, d_model: int, heads: int):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = d_model // heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.last_attn = None

    def _split_heads(self, x: mx.array, n_tokens: int) -> mx.array:
        B = x.shape[0]
        return x.reshape(B, n_tokens, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, F, d_model) -- F feature-tokens
        B, F, _ = x.shape
        q = self._split_heads(self.q_proj(x), F)
        k = self._split_heads(self.k_proj(x), F)
        v = self._split_heads(self.v_proj(x), F)
        scale = self.head_dim**-0.5
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, heads, F, F)
        weights = mx.softmax(scores, axis=-1)
        # Stored as a numpy array, NOT an mx.array: an mx.array attribute set
        # during __call__ is swept into the MLX module's trainable-parameter
        # tree (verified empirically) and would get spurious optimizer
        # updates every step. numpy is invisible to that tree, still gives
        # `.shape`, and is refreshed on every forward pass anyway.
        self.last_attn = np.array(mx.mean(weights, axis=(0, 1)))
        out = weights @ v  # (B, heads, F, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, F, self.heads * self.head_dim)
        return self.out_proj(out)


class _ITransformerBlock(nn.Module):
    """Pre-LN block: LN -> feature-token attention -> residual, LN -> FFN(GELU) -> residual."""

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _ITransformerAttention(d_model, heads)
        self.drop_a = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model * 2)
        self.ff2 = nn.Linear(d_model * 2, d_model)
        self.drop_f = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        res = x
        h = self.attn(self.ln1(x))
        h = self.drop_a(h)
        x = res + h

        res = x
        h = self.ln2(x)
        h = nn.gelu(self.ff1(h))
        h = self.drop_f(h)
        h = self.ff2(h)
        return res + h


class _ITransformerModel(nn.Module):
    """iTransformer: tokens = FEATURES (research R13 / Part B ref [7]).

    Transpose (B,T,F)->(B,F,T); embed each feature's length-T series with
    Linear(T, d_model) (no positional encoding -- the tokens are unordered
    features); `layers` pre-LN blocks of feature-token attention; mean-pool
    over tokens -> Linear -> softmax. `last_attention` (the final block's
    mean-over-heads/batch attention matrix, shape (num_features,
    num_features)) is exposed as a numpy array for the secondary
    (attention-based) importance signal.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        d_model: int,
        heads: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embed = nn.Linear(seq_len, d_model)
        self.blocks = [_ITransformerBlock(d_model, heads, dropout) for _ in range(layers)]
        self.out = nn.Linear(d_model, num_classes)
        self.last_attention = None

    def __call__(self, x: mx.array) -> mx.array:
        h = x.transpose(0, 2, 1)  # (B, F, T)
        h = self.embed(h)  # (B, F, d_model)
        for block in self.blocks:
            h = block(h)
            self.last_attention = block.attn.last_attn
        h = mx.mean(h, axis=1)  # mean-pool over feature tokens
        return mx.softmax(self.out(h), axis=-1)


class NNNClassifierMLX_ITransformer(MLXClassifierNary):
    """ITRANSFORMER registry type (research R13 / Part B ref [7]): feature-as-
    token attention. `hp` keys: `d_model` (64), `heads` (4), `layers` (2),
    `dropout` (0.25). Exposes `model.last_attention` (numpy, shape
    (num_features, num_features)) after a forward pass."""

    is_trained = False
    clean_data_required = False
    hp: dict = {}

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        d_model = self.hp.get("d_model", 64)
        heads = self.hp.get("heads", 4)
        layers = self.hp.get("layers", 2)
        dropout = self.hp.get("dropout", 0.25)
        return _ITransformerModel(
            seq_len, num_features, self.num_classes, d_model, heads, layers, dropout
        )


# -----------------------------------------------------------------------
# NNNClassifierMLX_XLSTM  (XLSTM registry type, research R13 / Part B ref [5])
# -----------------------------------------------------------------------


class _MLSTMBlock(nn.Module):
    """One mLSTM block (Beck et al. 2024, xLSTM): per-head exponential
    input/forget gating with a matrix memory, a log-space stabiliser state to
    keep the exponentials finite, sequential scan over time (mirrors the
    Python-loop-over-`t` + `mx.stack` pattern the existing Mamba (v1) port in
    this file uses for its own sequential scan), residual + LayerNorm.

    Per head, per time step t:
      q_t, k_t, v_t   = per-head linear projections of the block input
                         (k scaled by 1/sqrt(head_dim))
      i~_t, f~_t      = scalar (per head) input/forget gate pre-activations
      log_f_t         = log(sigmoid(f~_t))              -- bounded <= 0
      m_t             = max(log_f_t + m_{t-1}, i~_t)     -- stabiliser (log-space max trick)
      i_t             = exp(i~_t - m_t)
      f_t             = exp(log_f_t + m_{t-1} - m_t)
      C_t             = f_t * C_{t-1} + i_t * outer(v_t, k_t)
      n_t             = f_t * n_{t-1} + i_t * k_t
      h~_t            = (C_t @ q_t) / max(|n_t . q_t|, 1)
      o_t             = sigmoid(Linear(x_t))
      h_t             = o_t * h~_t

    Numerical safety: `i~_t` is clipped to +/-`_CLIP` before use; the exponent
    arguments actually passed to `exp` (`i~_t - m_t` and `log_f_t + m_{t-1} -
    m_t`) are always <= 0 by construction of `m_t` (a running max), so `exp`
    can never overflow regardless of input magnitude.
    """

    _CLIP = 30.0

    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden, hidden)
        self.i_proj = nn.Linear(hidden, heads)
        self.f_proj = nn.Linear(hidden, heads)
        self.out_proj = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, H = x.shape
        heads, d = self.heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, heads, d).transpose(0, 2, 1, 3)
        k = (self.k_proj(x) * (d**-0.5)).reshape(B, T, heads, d).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, heads, d).transpose(0, 2, 1, 3)
        o = mx.sigmoid(self.o_proj(x)).reshape(B, T, heads, d).transpose(0, 2, 1, 3)

        i_tilde = mx.clip(self.i_proj(x), -self._CLIP, self._CLIP).transpose(0, 2, 1)
        f_tilde = self.f_proj(x).transpose(0, 2, 1)  # (B, heads, T)
        log_f = -nn.softplus(-f_tilde)  # log(sigmoid(f_tilde)), <= 0

        C = mx.zeros((B, heads, d, d))
        n = mx.zeros((B, heads, d))
        m = mx.full((B, heads), -self._CLIP)

        hs = []
        for t in range(T):
            f_t = log_f[:, :, t]
            i_t = i_tilde[:, :, t]
            m_new = mx.maximum(f_t + m, i_t)
            i_gate = mx.exp(i_t - m_new)
            f_gate = mx.exp(f_t + m - m_new)
            m = m_new

            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]

            outer_vk = v_t[..., :, None] * k_t[..., None, :]  # (B, heads, d, d)
            C = f_gate[..., None, None] * C + i_gate[..., None, None] * outer_vk
            n = f_gate[..., None] * n + i_gate[..., None] * k_t

            numer = mx.sum(C * q_t[..., None, :], axis=-1)  # (B, heads, d)
            denom = mx.maximum(mx.abs(mx.sum(n * q_t, axis=-1)), 1.0)  # (B, heads)
            h_tilde = numer / denom[..., None]
            hs.append(o[:, :, t, :] * h_tilde)

        h = mx.stack(hs, axis=2)  # (B, heads, T, d)
        h = h.transpose(0, 2, 1, 3).reshape(B, T, H)
        h = self.drop(self.out_proj(h))
        return self.ln(x + h)


class _XLSTMModel(nn.Module):
    """Linear(F->hidden) -> `depth` `_MLSTMBlock`s -> last-timestep -> Linear -> softmax."""

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        hidden: int,
        depth: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, hidden)
        self.blocks = [_MLSTMBlock(hidden, heads, dropout) for _ in range(depth)]
        self.out = nn.Linear(hidden, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = h[:, -1, :]
        return mx.softmax(self.out(h), axis=-1)


class NNNClassifierMLX_XLSTM(MLXClassifierNary):
    """XLSTM registry type (research R13 / Part B ref [5]): mLSTM block (Beck et
    al. 2024) -- exponential gating + matrix memory. `hp` keys: `hidden` (64),
    `depth` (1), `heads` (4), `dropout` (0.25)."""

    is_trained = False
    clean_data_required = False
    hp: dict = {}

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        hidden = self.hp.get("hidden", 64)
        depth = self.hp.get("depth", 1)
        heads = self.hp.get("heads", 4)
        dropout = self.hp.get("dropout", 0.25)
        return _XLSTMModel(
            seq_len, num_features, self.num_classes, hidden, depth, heads, dropout
        )


# -----------------------------------------------------------------------
# NNNClassifierMLX_TransformerModern  (TRANSFORMER_MODERN registry type,
# research R13 / Part B refs [12] RMSNorm, [13] RoPE, [20] SwiGLU)
# -----------------------------------------------------------------------


class _SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: Linear(d,ffn) elementwise-gated by SiLU(Linear(d,ffn)),
    then Linear(ffn,d). Drop-in replacement for the plain GELU-MLP FFN used by
    `_TransformerFamBlock` (research R13 / Part B ref [20])."""

    def __init__(self, d_model: int, ffn_mult: int, dropout: float):
        super().__init__()
        ffn_dim = d_model * ffn_mult
        self.w = nn.Linear(d_model, ffn_dim)
        self.v = nn.Linear(d_model, ffn_dim)
        self.proj = nn.Linear(ffn_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.w(x) * nn.silu(self.v(x))
        h = self.drop(h)
        return self.proj(h)


class _TransformerModernBlock(nn.Module):
    """Pre-norm block: RMSNorm -> RoPE multi-head attention -> residual,
    RMSNorm -> SwiGLU FFN -> residual (research R13 / Part B refs [12,13,20])."""

    def __init__(self, d_model: int, heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = _RoPEMultiHeadAttention(d_model, heads)
        self.drop_a = nn.Dropout(dropout)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = _SwiGLUFFN(d_model, ffn_mult, dropout)

    def __call__(self, x: mx.array) -> mx.array:
        res = x
        h_in = self.norm1(x)
        h = self.attn(h_in, h_in, h_in)
        x = res + self.drop_a(h)

        res = x
        h = self.ffn(self.norm2(x))
        return res + h


class _TransformerModernModel(nn.Module):
    """Linear(F->d_model) -> `layers` `_TransformerModernBlock`s -> mean pool ->
    Linear -> softmax. No learned positional embedding -- position comes from
    RoPE inside every block's attention."""

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        d_model: int,
        heads: int,
        layers: int,
        ffn_mult: int,
        dropout: float,
    ):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        self.blocks = [
            _TransformerModernBlock(d_model, heads, ffn_mult, dropout) for _ in range(layers)
        ]
        self.out = nn.Linear(d_model, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.proj(x)
        for block in self.blocks:
            h = block(h)
        h = global_avg_pool_1d(h)
        return mx.softmax(self.out(h), axis=-1)


class NNNClassifierMLX_TransformerModern(MLXClassifierNary):
    """TRANSFORMER_MODERN registry type (research R13 / Part B refs [12,13,20]):
    `_TransformerFamModel`-style stack with `nn.RMSNorm` pre-norm, per-head RoPE
    attention (`_RoPEMultiHeadAttention`) and a SwiGLU FFN. `hp` keys: `d_model`
    (64), `heads` (4), `layers` (2), `ffn_mult` (2), `dropout` (0.25)."""

    is_trained = False
    clean_data_required = False
    hp: dict = {}

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        d_model = self.hp.get("d_model", 64)
        heads = self.hp.get("heads", 4)
        layers = self.hp.get("layers", 2)
        ffn_mult = self.hp.get("ffn_mult", 2)
        dropout = self.hp.get("dropout", 0.25)
        return _TransformerModernModel(
            seq_len, num_features, self.num_classes, d_model, heads, layers, ffn_mult, dropout
        )


# -----------------------------------------------------------------------
# NNNClassifierMLX_PatchTST  (PATCHTST registry type, research R13 / Part B ref [6])
# -----------------------------------------------------------------------


class _PatchTSTModel(nn.Module):
    """Channel-independent patch transformer (PatchTST, research R13 / Part B
    ref [6]). Each feature channel's length-T series is unfolded into
    overlapping patches, patch-embedded with a shared Linear(patch_len,
    d_model) + learned positional embedding over patch index, run through a
    shared pre-LN transformer encoder (`_TransformerFamBlock`, reused as-is --
    plain `nn.MultiHeadAttention`, no extra positional encoding inside the
    block since position is already carried by the patch embedding),
    mean-pooled over patches per channel, then channels are concatenated and
    read out with a single Linear -> softmax.

    Raises `ValueError` if `seq_len < 4 * patch_len` (too few patches for the
    patching mechanism to mean anything -- production seq_len is 16, so this
    member is only meaningful at larger window lengths).
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_classes: int,
        patch_len: int,
        stride: int,
        d_model: int,
        heads: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        if seq_len < 4 * patch_len:
            raise ValueError(
                f"PATCHTST requires seq_len >= 4 * patch_len ({4 * patch_len}), "
                f"got seq_len={seq_len} (patch_len={patch_len})"
            )
        self.patch_len = patch_len
        self.stride = stride
        self.num_features = num_features
        self.d_model = d_model
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_emb = TransformerPositionalEmbeddingMLX(self.num_patches, d_model)
        ffn_mult = 2
        self.blocks = [
            _TransformerFamBlock(
                nn.MultiHeadAttention(dims=d_model, num_heads=heads), d_model, ffn_mult, dropout
            )
            for _ in range(layers)
        ]
        self.out = nn.Linear(d_model * num_features, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, F = x.shape
        xt = x.transpose(0, 2, 1)  # (B, F, T)
        patches = mx.stack(
            [
                xt[:, :, i * self.stride : i * self.stride + self.patch_len]
                for i in range(self.num_patches)
            ],
            axis=2,
        )  # (B, F, num_patches, patch_len)
        patches = patches.reshape(B * F, self.num_patches, self.patch_len)

        h = self.patch_embed(patches)  # (B*F, num_patches, d_model)
        h = self.pos_emb(h)
        for block in self.blocks:
            h = block(h)
        h = mx.mean(h, axis=1)  # mean-pool over patches -> (B*F, d_model)
        h = h.reshape(B, F * self.d_model)  # concatenate channels
        return mx.softmax(self.out(h), axis=-1)


class NNNClassifierMLX_PatchTST(MLXClassifierNary):
    """PATCHTST registry type (research R13 / Part B ref [6]): channel-independent
    patching, shared transformer encoder over patch tokens. `hp` keys:
    `patch_len` (16), `stride` (8), `d_model` (64), `heads` (4), `layers` (2),
    `dropout` (0.25). Requires `seq_len >= 4 * patch_len` -- raises
    `ValueError` otherwise."""

    is_trained = False
    clean_data_required = False
    hp: dict = {}

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        patch_len = self.hp.get("patch_len", 16)
        stride = self.hp.get("stride", 8)
        d_model = self.hp.get("d_model", 64)
        heads = self.hp.get("heads", 4)
        layers = self.hp.get("layers", 2)
        dropout = self.hp.get("dropout", 0.25)
        if seq_len < 4 * patch_len:
            raise ValueError(
                f"PATCHTST requires seq_len >= 4 * patch_len ({4 * patch_len}), "
                f"got seq_len={seq_len} (patch_len={patch_len})"
            )
        return _PatchTSTModel(
            seq_len,
            num_features,
            self.num_classes,
            patch_len,
            stride,
            d_model,
            heads,
            layers,
            dropout,
        )


# -----------------------------------------------------------------------
# ClassifierTypeMLX enum + factory
# -----------------------------------------------------------------------


class ClassifierTypeMLX(Enum):
    LSTM = NNNClassifierMLX_LSTM
    LSTM2 = NNNClassifierMLX_LSTM2
    GRU = NNNClassifierMLX_GRU
    CNN = NNNClassifierMLX_CNN
    MLP = NNNClassifierMLX_MLP
    Attention = NNNClassifierMLX_Attention
    Multihead = NNNClassifierMLX_Multihead
    Transformer = NNNClassifierMLX_Transformer
    Wavenet = NNNClassifierMLX_Wavenet
    Mamba = NNNClassifierMLX_Mamba
    Mamba2 = NNNClassifierMLX_Mamba2
    TSMamba = NNNClassifierMLX_TSMamba
    LSTM_KAN = NNNClassifierMLX_LSTM_KAN
    LSTM_INJIT = NNNClassifierMLX_LSTM_InJit
    LSTM_NOISY = NNNClassifierMLX_LSTM_Noisy
    LSTM_PONDER = NNNClassifierMLX_LSTM_Ponder
    LSTM_W64 = NNNClassifierMLX_LSTM_W64
    LSTM_D10 = NNNClassifierMLX_LSTM_D10
    LSTM_POOL = NNNClassifierMLX_LSTM_Pool
    LSTM_K5 = NNNClassifierMLX_LSTM_K5
    LSTM_NOCONV = NNNClassifierMLX_LSTM_NoConv
    LSTM_RAW = NNNClassifierMLX_LSTM_Raw
    LSTM_NORESIZE = NNNClassifierMLX_LSTM_NoResize
    LSTM_NORES = NNNClassifierMLX_LSTM_NoRes
    RIDGE = NNNClassifierMLX_Ridge
    ROCKET = NNNClassifierMLX_Rocket
    LSTM_FAM = NNNClassifierMLX_LSTM_Fam
    GRU_FAM = NNNClassifierMLX_GRU_Fam
    TCN_FAM = NNNClassifierMLX_TCN_Fam
    TRANSFORMER_FAM = NNNClassifierMLX_Transformer_Fam
    MLP_FAM = NNNClassifierMLX_MLP_Fam
    TSMIXER = NNNClassifierMLX_TSMixer
    ITRANSFORMER = NNNClassifierMLX_ITransformer
    XLSTM = NNNClassifierMLX_XLSTM
    TRANSFORMER_MODERN = NNNClassifierMLX_TransformerModern
    PATCHTST = NNNClassifierMLX_PatchTST


def create_classifier_mlx(
    clf_type: ClassifierTypeMLX,
    pair: str,
    nfeatures: int,
    seq_len: int,
    nclasses: int,
    tag: str = "",
):
    """
    Factory function — drop-in replacement for NNNClassifier.create_classifier().

    Returns (classifier, name) tuple with identical interface to the Keras version.
    """
    clf_name = str(clf_type).split(".")[-1]
    clf = clf_type.value(pair, seq_len, nfeatures, tag=tag)
    clf.num_classes = nclasses
    return clf, clf_name


class MLXClassifierMixin:
    """Shared MLX classifier construction for NNNC strategy bases.

    Mixed in (before the family base) by the MLX single-task strategy bases
    so the MLX ``get_classifier`` lives in exactly one place instead of being
    copy-pasted into each base. Architecture selection is
    declarative: set ``classifier_type`` to a ``ClassifierTypeMLX`` value on
    the subclass (default LSTM). ``get_classifier_type`` is inherited from
    NNNCStrategy and returns ``self.classifier_type``.
    """

    classifier_type = ClassifierTypeMLX.LSTM

    def get_classifier(self, classifier_type, pair, seq_len, num_features):
        """Return the MLX classifier used for training/predicting."""
        if hasattr(mx, "metal") and mx.metal.is_available():
            clf, _ = create_classifier_mlx(
                classifier_type, pair, num_features, seq_len, 3
            )
        else:
            print(
                "ERROR: This strategy requires Apple's MLX package, and only runs on native Apple hardware"
            )
            clf = None
        return clf
