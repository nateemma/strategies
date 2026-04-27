from __future__ import annotations

# MLX multi-task classifier variants, mirroring NNMTClassifier.py.
#
# Each public class inherits ClassifierMLXMultiTask and provides one of:
#   * _make_common_layer  — overridden by "Normal" variants (LSTM, Transformer,
#                           CNN, GRU, Wavenet, Wavenet_Fast, Attention) to plug
#                           an architecture-specific module into the *shared*
#                           backbone.  The per-task heads stay at the default.
#   * _make_task_head     — overridden by "Multi_*" variants to plug an
#                           architecture-specific module into *each* of the
#                           six task heads.  The shared backbone stays at the
#                           default GRU.  Per-user request, the same head
#                           architecture is used for every task — there is no
#                           per-task branching like the TF version had.
#
# Usage (drop-in for NNMTClassifier.create_classifier):
#
#   from NNMT.NNMTClassifierMLX import create_classifier_mlx, ClassifierTypeMLX
#   clf, name = create_classifier_mlx(ClassifierTypeMLX.LSTM, pair, nfeatures, seq_len)
#   clf.set_model_path(path)
#   clf.train(df_train, df_test, train_results_dict, test_results_dict)
#   preds_dict = clf.predict(df)        # {"trading": ..., "regime": ..., ...}

import sys
from enum import Enum
from pathlib import Path
from typing import List

import mlx.core as mx
import mlx.nn as nn

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from utils.ClassifierMLXMultiTask import ClassifierMLXMultiTask, TASK_NAMES


# ---------------------------------------------------------------------------
# Module-level constants (mirror NNMTClassifier.py / NNNClassifierMLX.py)
# ---------------------------------------------------------------------------

DROPOUT = 0.2
MIN_FILTER = 32


def nearest_power_of_2(n: int) -> int:
    return 2 ** n.bit_length()


def global_avg_pool_1d(x: mx.array) -> mx.array:
    """Mean over the sequence dimension. (B, S, F) → (B, F)"""
    return mx.mean(x, axis=1)


# ---------------------------------------------------------------------------
# Helper layers (port of CausalConv1d + TransformerPositionalEmbedding from
# NNNClassifierMLX.py — duplicated locally to keep this file self-contained).
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """nn.Conv1d with causal (left-only) padding."""

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
        # x: (B, S, C)
        B, S, C = x.shape
        pad = mx.zeros((B, self.pad_width, C))
        x_padded = mx.concatenate([pad, x], axis=1)
        return self.conv(x_padded)


class TransformerPositionalEmbedding(nn.Module):
    """Learnable positional embedding."""

    def __init__(self, seq_len: int, embed_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        positions = mx.arange(self.seq_len)
        emb = self.pos_emb(positions)  # (S, embed_dim)
        return x + emb  # broadcast over batch


def _trim_seq(x: mx.array, target_len: int) -> mx.array:
    """Trim the temporal dimension to ``target_len`` (handles asymmetric Conv1d padding)."""
    if x.shape[1] == target_len:
        return x
    return x[:, :target_len, :]


# ---------------------------------------------------------------------------
# Default building blocks
# ---------------------------------------------------------------------------


class _DefaultGRUCommon(nn.Module):
    """Default common layer: GRU over (B, S, F)."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.gru = nn.GRU(num_filters, num_filters)
        self.ln = nn.LayerNorm(num_filters)
        self.drop = nn.Dropout(DROPOUT)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.gru(x)
        x = self.ln(x)
        x = self.drop(x)
        return x


class _DefaultTaskHead(nn.Module):
    """
    Default per-task head — uniform across all variants that don't override
    _make_task_head.  Pools the sequence, projects through two dense layers,
    and emits 3-class softmax.
    """

    def __init__(self, num_filters: int):
        super().__init__()
        nf = max(MIN_FILTER, nearest_power_of_2((num_filters // 2) + 1))
        self.fc1 = nn.Linear(num_filters, nf)
        self.drop1 = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(nf, 16)
        self.drop2 = nn.Dropout(DROPOUT)
        self.out = nn.Linear(16, 3)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, S, F)
        h = global_avg_pool_1d(x)  # (B, F)
        h = self.drop1(nn.elu(self.fc1(h)))
        h = self.drop2(nn.elu(self.fc2(h)))
        return mx.softmax(self.out(h), axis=-1)


# ---------------------------------------------------------------------------
# Backbone — composes resized_input + common-layer base + custom common +
# six task heads.  Mirrors NNMTClassifier_Base.create_model().
# ---------------------------------------------------------------------------


class _NNMTModel(nn.Module):
    """
    Architecture-agnostic backbone for the multi-task classifier.

    Forward path:
        Input (B, S, F)
        → Conv1d(2)+LN+Dropout         (resized_input, channels=num_filters)
        → Conv1d(5)+LN+Dropout × 2     (shared backbone preamble)
        → custom_common (subclass)     ((B, S, num_filters))
        → residual add with resized_input
        → six task heads (each → (B, 3) softmax)
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_filters: int,
        custom_common: nn.Module,
        task_heads: List[nn.Module],
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_filters = num_filters

        # Resize block
        self.resize_conv = nn.Conv1d(
            num_features, num_filters, kernel_size=2, padding=1
        )
        self.resize_ln = nn.LayerNorm(num_filters)
        self.resize_drop = nn.Dropout(DROPOUT)

        # Common-layer preamble: 2× Conv1d(5)
        self.common_conv1 = nn.Conv1d(
            num_filters, num_filters, kernel_size=5, padding=2
        )
        self.common_ln1 = nn.LayerNorm(num_filters)
        self.common_drop1 = nn.Dropout(DROPOUT)
        self.common_conv2 = nn.Conv1d(
            num_filters, num_filters, kernel_size=5, padding=2
        )
        self.common_ln2 = nn.LayerNorm(num_filters)
        self.common_drop2 = nn.Dropout(DROPOUT)

        # Custom common layer (architecture-specific)
        self.custom_common = custom_common

        # Per-task heads (one per TASK_NAMES entry, in fixed order)
        if len(task_heads) != len(TASK_NAMES):
            raise ValueError(
                f"task_heads must have {len(TASK_NAMES)} entries "
                f"(got {len(task_heads)})"
            )
        self.task_heads = list(task_heads)

    def __call__(self, x: mx.array):  # type: ignore[override]
        S = x.shape[1]

        # Resize
        x_r = self.resize_conv(x)
        x_r = _trim_seq(x_r, S)
        x_r = nn.relu(x_r)
        x_r = self.resize_ln(x_r)
        x_r = self.resize_drop(x_r)

        # Common preamble
        c = self.common_conv1(x_r)
        c = _trim_seq(c, S)
        c = nn.elu(c)
        c = self.common_ln1(c)
        c = self.common_drop1(c)

        c = self.common_conv2(c)
        c = _trim_seq(c, S)
        c = nn.elu(c)
        c = self.common_ln2(c)
        c = self.common_drop2(c)

        # Architecture-specific common layer
        c = self.custom_common(c)

        # Residual add (shapes match because every common layer keeps num_filters)
        c = c + x_r

        # Per-task heads → dict of (B, 3) softmax
        return {task: head(c) for task, head in zip(TASK_NAMES, self.task_heads)}


# ===========================================================================
# Common-layer variants — used by "Normal" classifier types
# Each takes (B, S, num_filters) and returns (B, S, num_filters).
# ===========================================================================


class _LSTMCommon(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        self.lstm = nn.LSTM(num_filters, num_filters)
        self.ln = nn.LayerNorm(num_filters)
        self.drop = nn.Dropout(DROPOUT)

    def __call__(self, x: mx.array) -> mx.array:
        out, _ = self.lstm(x)
        out = self.ln(out)
        out = self.drop(out)
        return out


class _TransformerCommon(nn.Module):
    """Three transformer encoder layers with residual+LN, output channels = num_filters."""

    def __init__(self, num_filters: int, num_heads: int = 8, ff_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(num_filters, num_filters)
        self.ln_in = nn.LayerNorm(num_filters)
        self.layers = []
        for _ in range(3):
            self.layers.append(
                {
                    "mha": nn.MultiHeadAttention(
                        dims=num_filters, num_heads=num_heads
                    ),
                    "ln1": nn.LayerNorm(num_filters),
                    "ffn1": nn.Linear(num_filters, ff_dim),
                    "drop_ffn": nn.Dropout(DROPOUT),
                    "ffn2": nn.Linear(ff_dim, num_filters),
                    "ln2": nn.LayerNorm(num_filters),
                }
            )
        # NOTE: dicts of submodules are exercised here; if a future MLX version
        # complains about parameter tracking, fall back to flat attributes.

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        x = self.ln_in(x)
        for blk in self.layers:
            attn_out = blk["mha"](x, x, x)
            x = x + attn_out
            x = blk["ln1"](x)
            ff = blk["ffn2"](blk["drop_ffn"](nn.relu(blk["ffn1"](x))))
            x = x + ff
            x = blk["ln2"](x)
        return x


class _CNNCommon(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm(num_filters)
        self.drop = nn.Dropout(DROPOUT)

    def __call__(self, x: mx.array) -> mx.array:
        S = x.shape[1]
        out = self.conv(x)
        out = _trim_seq(out, S)
        out = nn.elu(out)
        out = self.ln(out)
        out = self.drop(out)
        return out


class _GRUCommon(nn.Module):
    """Explicit GRU common layer (same as default, named for clarity)."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.gru = nn.GRU(num_filters, num_filters)
        self.ln = nn.LayerNorm(num_filters)
        self.drop = nn.Dropout(DROPOUT)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.gru(x)
        x = self.ln(x)
        x = self.drop(x)
        return x


class _WavenetCommon(nn.Module):
    """Stacked dilated causal convolutions with skip connections."""

    def __init__(self, num_filters: int, num_blocks: int = 2, num_dilations: int = 9):
        super().__init__()
        self.entry = CausalConv1d(num_filters, num_filters, kernel_size=2, dilation=1)

        self.blocks = []
        for _ in range(num_blocks):
            block = []
            rate = 1
            for _ in range(num_dilations):
                block.append(
                    {
                        "tanh": CausalConv1d(
                            num_filters, num_filters, kernel_size=2, dilation=rate
                        ),
                        "sig": CausalConv1d(
                            num_filters, num_filters, kernel_size=2, dilation=rate
                        ),
                        "skip": nn.Conv1d(
                            num_filters, num_filters, kernel_size=1
                        ),
                        "res": nn.Conv1d(
                            num_filters, num_filters, kernel_size=1
                        ),
                    }
                )
                rate *= 2
            self.blocks.append(block)
        self.block_lns = [nn.LayerNorm(num_filters) for _ in range(num_blocks)]

        self.post1 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.post2 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.drop = nn.Dropout(DROPOUT)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.entry(x)
        skips = []
        for block, ln in zip(self.blocks, self.block_lns):
            for unit in block:
                tanh_out = nn.tanh(unit["tanh"](x))
                sig_out = mx.sigmoid(unit["sig"](x))
                gated = tanh_out * sig_out
                skip = unit["skip"](gated)
                res = unit["res"](x)
                x = res + gated
                skips.append(skip)
            x = ln(x)

        merged = skips[0]
        for s in skips[1:]:
            merged = merged + s
        merged = nn.relu(merged)
        merged = nn.relu(self.post1(merged))
        merged = self.post2(merged)
        merged = self.drop(merged)
        return merged


class _WavenetFastCommon(nn.Module):
    """Fast Wavenet — fewer blocks, fewer dilations, last-3 skip aggregation."""

    def __init__(self, num_filters: int):
        super().__init__()
        n = min(num_filters, 64)
        self.proj_in = nn.Conv1d(num_filters, n, kernel_size=1)
        self.entry = CausalConv1d(n, n, kernel_size=2, dilation=1)
        # 1 block × 6 dilations (1, 2, 4, 8, 16, 32)
        self.units = []
        rate = 1
        for _ in range(6):
            self.units.append(
                {
                    "tanh": CausalConv1d(n, n, kernel_size=2, dilation=rate),
                    "sig":  CausalConv1d(n, n, kernel_size=2, dilation=rate),
                    "skip": nn.Conv1d(n, n, kernel_size=1),
                    "res":  nn.Conv1d(n, n, kernel_size=1),
                }
            )
            rate = min(rate * 2, 32)
        self.ln = nn.LayerNorm(n)

        self.post1 = nn.Conv1d(n, n, kernel_size=1)
        # Project back to num_filters for residual add at the wrapper
        self.proj_out = nn.Conv1d(n, num_filters, kernel_size=1)
        self.drop = nn.Dropout(DROPOUT)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj_in(x)
        x = self.entry(x)
        skips = []
        for unit in self.units:
            tanh_out = nn.tanh(unit["tanh"](x))
            sig_out = mx.sigmoid(unit["sig"](x))
            gated = tanh_out * sig_out
            skip = unit["skip"](gated)
            res = unit["res"](x)
            x = res + gated
            skips.append(skip)
        x = self.ln(x)

        # Aggregate only the last 3 skip connections
        kept = skips[-3:]
        merged = kept[0]
        for s in kept[1:]:
            merged = merged + s
        merged = nn.relu(merged)
        merged = nn.relu(self.post1(merged))
        merged = self.proj_out(merged)
        merged = self.drop(merged)
        return merged


class _AttentionCommon(nn.Module):
    """Single self-attention block with residual + LayerNorm."""

    def __init__(self, num_filters: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiHeadAttention(dims=num_filters, num_heads=num_heads)
        self.ln = nn.LayerNorm(num_filters)

    def __call__(self, x: mx.array) -> mx.array:
        attn = self.mha(x, x, x)
        x = x + attn
        x = self.ln(x)
        return x


# ===========================================================================
# Task-head variants — used by "Multi_*" classifier types
# Each takes (B, S, num_filters) and returns (B, 3) softmax.
# Per user request: applied uniformly across all six tasks (no per-task
# branching like the TF version had).
# ===========================================================================


class _LSTMHead(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        nf = max(MIN_FILTER, nearest_power_of_2((num_filters // 2) + 1))
        self.cnn = nn.Conv1d(num_filters, nf, kernel_size=5, padding=2)
        self.ln_cnn = nn.LayerNorm(nf)
        self.drop_cnn = nn.Dropout(DROPOUT)

        self.lstm = nn.LSTM(nf, nf)
        self.ln_lstm = nn.LayerNorm(nf)
        self.drop_lstm = nn.Dropout(DROPOUT)

        self.fc = nn.Linear(nf, nf)
        self.drop_fc = nn.Dropout(DROPOUT)
        self.out = nn.Linear(nf, 3)

    def __call__(self, x: mx.array) -> mx.array:
        S = x.shape[1]
        c = nn.elu(self.cnn(x))
        c = _trim_seq(c, S)
        c = self.ln_cnn(c)
        c = self.drop_cnn(c)

        out, _ = self.lstm(c)
        last = out[:, -1, :]  # (B, nf)
        last = self.ln_lstm(last)
        last = self.drop_lstm(last)

        h = nn.elu(self.fc(last))
        h = self.drop_fc(h)
        return mx.softmax(self.out(h), axis=-1)


class _GRUHead(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        half = num_filters // 2
        self.causal = CausalConv1d(num_filters, num_filters, kernel_size=2)
        self.gru = nn.GRU(num_filters, half)
        self.drop1 = nn.Dropout(DROPOUT)
        self.fc1 = nn.Linear(half, half)
        self.drop2 = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(half, 16)
        self.drop3 = nn.Dropout(DROPOUT)
        self.out = nn.Linear(16, 3)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.tanh(self.causal(x))
        x = self.gru(x)
        x = self.drop1(x)
        h = global_avg_pool_1d(x)  # (B, half)
        h = nn.tanh(self.fc1(h))
        h = self.drop2(h)
        h = nn.tanh(self.fc2(h))
        h = self.drop3(h)
        return mx.softmax(self.out(h), axis=-1)


class _CNNHead(nn.Module):
    """Simplified Wavenet-style stack."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.layers = []
        rates = [1, 2, 4, 8, 1, 2, 4, 8]
        in_ch = num_filters
        for r in rates:
            self.layers.append(
                {
                    "conv": CausalConv1d(in_ch, num_filters, kernel_size=2, dilation=r),
                    "drop": nn.Dropout(DROPOUT),
                }
            )
            in_ch = num_filters
        self.fc1 = nn.Linear(num_filters, num_filters // 2)
        self.drop1 = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(num_filters // 2, 16)
        self.drop2 = nn.Dropout(DROPOUT)
        self.out = nn.Linear(16, 3)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = nn.tanh(layer["conv"](x))
            x = layer["drop"](x)
        h = global_avg_pool_1d(x)
        h = nn.tanh(self.fc1(h))
        h = self.drop1(h)
        h = nn.tanh(self.fc2(h))
        h = self.drop2(h)
        return mx.softmax(self.out(h), axis=-1)


class _WavenetHead(nn.Module):
    """Lighter Wavenet head with avg+max pooling concat, used uniformly per task."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.layers = []
        # 8 dilated convs, layer-norm every 1, residual every 2
        rates = [1, 2, 4, 8] * 2
        in_ch = num_filters
        self.lns = []
        for i, r in enumerate(rates):
            self.layers.append(
                CausalConv1d(in_ch, num_filters, kernel_size=2, dilation=r)
            )
            self.lns.append(nn.LayerNorm(num_filters))
            in_ch = num_filters

        self.fc1 = nn.Linear(2 * num_filters, 2 * num_filters)
        self.drop1 = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(2 * num_filters, 16)
        self.drop2 = nn.Dropout(DROPOUT)
        self.out = nn.Linear(16, 3)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        for i, (conv, ln) in enumerate(zip(self.layers, self.lns)):
            new = nn.relu(conv(x))
            new = ln(new)
            if i % 2 == 1:
                x = new + residual
                residual = x
            else:
                x = new
        avg = global_avg_pool_1d(x)
        mx_pool = mx.max(x, axis=1)
        cat = mx.concatenate([avg, mx_pool], axis=-1)
        h = nn.tanh(self.fc1(cat))
        h = self.drop1(h)
        h = nn.tanh(self.fc2(h))
        h = self.drop2(h)
        return mx.softmax(self.out(h), axis=-1)


class _MLPHead(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        self.fc1 = nn.Linear(num_filters, num_filters)
        self.drop1 = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(num_filters, 16)
        self.drop2 = nn.Dropout(DROPOUT)
        self.out = nn.Linear(16, 3)

    def __call__(self, x: mx.array) -> mx.array:
        h = global_avg_pool_1d(x)
        h = nn.tanh(self.fc1(h))
        h = self.drop1(h)
        h = nn.tanh(self.fc2(h))
        h = self.drop2(h)
        return mx.softmax(self.out(h), axis=-1)


class _AttentionHead(nn.Module):
    """Multi-head self-attention block + FFN, then pool."""

    def __init__(self, num_filters: int, num_heads: int = 8):
        super().__init__()
        self.mha = nn.MultiHeadAttention(dims=num_filters, num_heads=num_heads)
        self.ln1 = nn.LayerNorm(num_filters)
        self.ffn1 = nn.Linear(num_filters, num_filters)
        self.drop_ffn = nn.Dropout(DROPOUT)
        self.ffn2 = nn.Linear(num_filters, 2 * num_filters)
        self.ln2 = nn.LayerNorm(2 * num_filters)

        self.fc = nn.Linear(2 * num_filters, num_filters // 2)
        self.drop_fc = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(num_filters // 2, 16)
        self.drop2 = nn.Dropout(DROPOUT)
        self.out = nn.Linear(16, 3)

    def __call__(self, x: mx.array) -> mx.array:
        attn = self.mha(x, x, x)
        x = self.ln1(x + attn)
        ff = self.ffn2(self.drop_ffn(nn.tanh(self.ffn1(x))))
        x = self.ln2(ff)
        h = global_avg_pool_1d(x)
        h = nn.tanh(self.fc(h))
        h = self.drop_fc(h)
        h = nn.tanh(self.fc2(h))
        h = self.drop2(h)
        return mx.softmax(self.out(h), axis=-1)


class _TransformerHead(nn.Module):
    """Stacked transformer encoder blocks per head + pooling."""

    def __init__(
        self,
        num_filters: int,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 64,
    ):
        super().__init__()
        self.pos_emb = TransformerPositionalEmbedding(seq_len=64, embed_dim=num_filters)
        # NOTE: seq_len=64 is a generous upper bound; the Embedding only stores
        # ``seq_len`` rows but indexing is bounded by the actual sequence length.
        # Strategies should not feed sequences longer than 64 — extend if needed.

        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(
                {
                    "mha": nn.MultiHeadAttention(
                        dims=num_filters, num_heads=num_heads
                    ),
                    "ln1": nn.LayerNorm(num_filters),
                    "ffn1": nn.Linear(num_filters, ff_dim),
                    "drop": nn.Dropout(DROPOUT),
                    "ffn2": nn.Linear(ff_dim, num_filters),
                    "ln2": nn.LayerNorm(num_filters),
                }
            )

        self.fc = nn.Linear(num_filters, num_filters // 2)
        self.drop_fc = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(num_filters // 2, 16)
        self.drop2 = nn.Dropout(DROPOUT)
        self.out = nn.Linear(16, 3)

    def __call__(self, x: mx.array) -> mx.array:
        # Trim the positional embedding to the actual sequence length.
        S = x.shape[1]
        pe = self.pos_emb.pos_emb(mx.arange(S))  # (S, F)
        x = x + pe

        for blk in self.blocks:
            attn = blk["mha"](x, x, x)
            x = blk["ln1"](x + attn)
            ff = blk["ffn2"](blk["drop"](nn.relu(blk["ffn1"](x))))
            x = blk["ln2"](x + ff)

        h = global_avg_pool_1d(x)
        h = nn.tanh(self.fc(h))
        h = self.drop_fc(h)
        h = nn.tanh(self.fc2(h))
        h = self.drop2(h)
        return mx.softmax(self.out(h), axis=-1)


# ---------------------------------------------------------------------------
# Base ClassifierMLXMultiTask subclass — owns the factory hooks
# ---------------------------------------------------------------------------


class NNMTClassifierMLX_Base(ClassifierMLXMultiTask):
    """
    Base multi-task MLX classifier.  Subclasses override exactly one of:
      * _make_common_layer   (Normal variants — change the shared backbone)
      * _make_task_head      (Multi_* variants — change every task head)
    """

    is_trained = False
    clean_data_required = False

    # ---------- factory hooks ----------

    def _make_common_layer(self, num_filters: int) -> nn.Module:
        """Default common layer: GRU.  Multi_* variants leave this alone."""
        return _DefaultGRUCommon(num_filters)

    def _make_task_head(self, num_filters: int) -> nn.Module:
        """Default task head.  Normal variants leave this alone."""
        return _DefaultTaskHead(num_filters)

    # ---------- create_model ----------

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        num_filters = max(MIN_FILTER, nearest_power_of_2(num_features - 1))
        custom_common = self._make_common_layer(num_filters)
        task_heads = [self._make_task_head(num_filters) for _ in TASK_NAMES]
        return _NNMTModel(seq_len, num_features, num_filters, custom_common, task_heads)


# ===========================================================================
# Normal variants — override _make_common_layer
# ===========================================================================


class NNMTClassifierMLX_LSTM(NNMTClassifierMLX_Base):
    def _make_common_layer(self, num_filters: int) -> nn.Module:
        return _LSTMCommon(num_filters)


class NNMTClassifierMLX_Transformer(NNMTClassifierMLX_Base):
    def _make_common_layer(self, num_filters: int) -> nn.Module:
        return _TransformerCommon(num_filters)


class NNMTClassifierMLX_CNN(NNMTClassifierMLX_Base):
    def _make_common_layer(self, num_filters: int) -> nn.Module:
        return _CNNCommon(num_filters)


class NNMTClassifierMLX_GRU(NNMTClassifierMLX_Base):
    def _make_common_layer(self, num_filters: int) -> nn.Module:
        return _GRUCommon(num_filters)


class NNMTClassifierMLX_Wavenet(NNMTClassifierMLX_Base):
    def _make_common_layer(self, num_filters: int) -> nn.Module:
        return _WavenetCommon(num_filters)


class NNMTClassifierMLX_Wavenet_Fast(NNMTClassifierMLX_Base):
    def _make_common_layer(self, num_filters: int) -> nn.Module:
        return _WavenetFastCommon(num_filters)


class NNMTClassifierMLX_Attention(NNMTClassifierMLX_Base):
    def _make_common_layer(self, num_filters: int) -> nn.Module:
        return _AttentionCommon(num_filters)


# ===========================================================================
# Multi_* variants — override _make_task_head
# All six task heads share the same architecture (no per-task branching).
# ===========================================================================


class NNMTClassifierMLX_Multi_LSTM(NNMTClassifierMLX_Base):
    def _make_task_head(self, num_filters: int) -> nn.Module:
        return _LSTMHead(num_filters)


class NNMTClassifierMLX_Multi_GRU(NNMTClassifierMLX_Base):
    def _make_task_head(self, num_filters: int) -> nn.Module:
        return _GRUHead(num_filters)


class NNMTClassifierMLX_Multi_CNN(NNMTClassifierMLX_Base):
    def _make_task_head(self, num_filters: int) -> nn.Module:
        return _CNNHead(num_filters)


class NNMTClassifierMLX_Multi_Wavenet(NNMTClassifierMLX_Base):
    def _make_task_head(self, num_filters: int) -> nn.Module:
        return _WavenetHead(num_filters)


class NNMTClassifierMLX_Multi_MLP(NNMTClassifierMLX_Base):
    def _make_task_head(self, num_filters: int) -> nn.Module:
        return _MLPHead(num_filters)


class NNMTClassifierMLX_Multi_Attention(NNMTClassifierMLX_Base):
    def _make_task_head(self, num_filters: int) -> nn.Module:
        return _AttentionHead(num_filters)


class NNMTClassifierMLX_Multi_Transformer(NNMTClassifierMLX_Base):
    def _make_task_head(self, num_filters: int) -> nn.Module:
        return _TransformerHead(num_filters)


# ---------------------------------------------------------------------------
# ClassifierTypeMLX enum + factory
# ---------------------------------------------------------------------------


class ClassifierTypeMLX(Enum):
    LSTM = NNMTClassifierMLX_LSTM
    Transformer = NNMTClassifierMLX_Transformer
    CNN = NNMTClassifierMLX_CNN
    GRU = NNMTClassifierMLX_GRU
    Wavenet = NNMTClassifierMLX_Wavenet
    Wavenet_Fast = NNMTClassifierMLX_Wavenet_Fast
    Attention = NNMTClassifierMLX_Attention
    Multi_LSTM = NNMTClassifierMLX_Multi_LSTM
    Multi_GRU = NNMTClassifierMLX_Multi_GRU
    Multi_CNN = NNMTClassifierMLX_Multi_CNN
    Multi_Wavenet = NNMTClassifierMLX_Multi_Wavenet
    Multi_MLP = NNMTClassifierMLX_Multi_MLP
    Multi_Attention = NNMTClassifierMLX_Multi_Attention
    Multi_Transformer = NNMTClassifierMLX_Multi_Transformer


def create_classifier_mlx(
    clf_type: ClassifierTypeMLX,
    pair: str,
    nfeatures: int,
    seq_len: int,
    *,
    tag: str = "",
):
    """
    Factory function — drop-in replacement for NNMTClassifier.create_classifier().

    Returns (classifier, name) tuple with the same interface as the Keras
    version.  ``tag`` is keyword-only — the multi-task classifier has no
    ``nclasses`` parameter (each task head is fixed at 3-way softmax), so a
    stray 5th positional argument should fail loudly rather than silently
    binding to ``tag`` and producing a confusing TypeError downstream.
    """
    clf_name = str(clf_type).split(".")[-1]
    clf = clf_type.value(pair, seq_len, nfeatures, tag=tag)
    return clf, clf_name
