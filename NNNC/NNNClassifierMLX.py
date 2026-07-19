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
