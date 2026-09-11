"""Long-only exposure policy (T022, T028).

FR-011  same normalised feature sequence and window length the control uses (seq_len=16),
        pooled across the whitelist
FR-012  smallest thing that can work: single-layer LSTM(64). NOT a transformer.
FR-013  both turnover controls: a learned no-trade band, and a plain head with an explicit
        |dw| penalty. Both are run and logged as separate trials.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

HIDDEN = 64
SEQ_LEN = 16


@dataclass(frozen=True)
class PolicyConfig:
    n_features: int
    hidden: int = HIDDEN
    seq_len: int = SEQ_LEN
    turnover_control: str = "band"        # "band" | "penalty"
    band_width: float = 0.10              # initial half-width when control == "band"

    def describe(self) -> dict:
        return {"kind": "lstm", "hidden": self.hidden, "seq_len": self.seq_len,
                "turnover_control": self.turnover_control,
                "band_width": self.band_width if self.turnover_control == "band" else None}


class ExposurePolicy(nn.Module):
    """features -> exposure in [0, 1], optionally with a learned no-trade band.

    The band variant emits lo/hi around the raw exposure and holds the previous position
    inside them (Imaki 2021): inaction is the default, and the clamp's active branch keeps
    a gradient. The penalty variant emits exposure directly and pays lambda*|dw| in the loss.
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(cfg.n_features, cfg.hidden)
        self.head = nn.Linear(cfg.hidden, 1)
        if cfg.turnover_control == "band":
            self.band = nn.Linear(cfg.hidden, 2)

    def raw(self, x: mx.array) -> tuple[mx.array, mx.array | None]:
        """x: (T, F) or (B, T, F) -> target exposure (and band half-widths)."""
        h, _ = self.lstm(x)
        w = mx.sigmoid(self.head(h)).squeeze(-1)
        if self.cfg.turnover_control != "band":
            return w, None
        # softplus keeps half-widths positive; scaled so the initial band is ~band_width
        b = nn.softplus(self.band(h)) * self.cfg.band_width
        return w, b

    def __call__(self, x: mx.array) -> mx.array:
        """Realised exposure path, applying the no-trade band sequentially if configured."""
        w, b = self.raw(x)
        if b is None:
            return w
        lo = mx.clip(w - b[..., 0], 0.0, 1.0)
        hi = mx.clip(w + b[..., 1], 0.0, 1.0)
        return band_hold(w, lo, hi)


def band_hold(w: mx.array, lo: mx.array, hi: mx.array) -> mx.array:
    """Hold the previous exposure while it sits inside [lo, hi]; otherwise move to the edge.

    Sequential by construction (each step depends on the last), so it is written as a scan.
    The clamp is differentiable on its active branch, which is what makes the band trainable
    rather than a post-hoc threshold.
    """
    T = w.shape[-1]
    prev = mx.zeros(w.shape[:-1]) if w.ndim > 1 else mx.array(0.0)
    out = []
    for t in range(T):
        lo_t, hi_t = lo[..., t], hi[..., t]
        held = mx.minimum(mx.maximum(prev, lo_t), hi_t)
        out.append(held)
        prev = held
    return mx.stack(out, axis=-1)


# ---------------------------------------------------------------------------
# Classifier-encoder policy (feature 010, Arm A/B/C).
# ---------------------------------------------------------------------------

class ClassifierEncoderPolicy(nn.Module):
    """Mirrors NNNClassifierMLX_LSTM.encode() tensor-for-tensor, plus an exposure head.

    Brief 01's policy is a plain LSTM(64); the classifier is
    resize -> conv1 -> ln_conv -> residual -> LSTM(32) -> ln_lstm. Those weights cannot
    load into that policy, so all three of brief 02's arms use THIS architecture: Arm A
    inherits its weights, B and C initialise it randomly. That keeps the warm-start a
    literal inheritance rather than a lossy projection, and keeps the arms comparable.

    Tensor names match the classifier exactly so `load_encoder` is a plain assignment with
    no renaming — a rename is where a silent partial load would hide.

    NOTE: encode() consumes a WINDOW and returns one latent per window (the LSTM's last
    timestep), not a per-bar sequence. A per-bar exposure therefore needs one window per
    bar. Measured: that is 0.6x the cost of brief 01's streaming pass, because MLX
    parallelises across the window batch — but the window tensor is 2.14 GB panel-wide, so
    windows MUST be built per chunk.
    """

    def __init__(self, n_features: int, num_filters: int = 32,
                 turnover_control: str = "band", band_width: float = 0.10):
        super().__init__()
        self.num_filters = num_filters
        self.turnover_control = turnover_control
        self.band_width = band_width
        # --- inherited from the classifier (names must match exactly) ---
        self.resize = nn.Linear(n_features, num_filters)
        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size=2, padding=1)
        self.ln_conv = nn.LayerNorm(num_filters)
        self.lstm = nn.LSTM(num_filters, num_filters)
        self.ln_lstm = nn.LayerNorm(num_filters)
        # --- new: replaces the classifier's decode() (bottleneck -> out -> softmax) ---
        self.head = nn.Linear(num_filters, 1)
        if turnover_control == "band":
            self.band = nn.Linear(num_filters, 2)

    ENCODER_TENSORS = ("resize", "conv1", "ln_conv", "lstm", "ln_lstm")

    def encode(self, xw: mx.array) -> mx.array:
        """(B, seq, features) -> latent (B, num_filters). Mirrors the classifier."""
        x_r = nn.relu(self.resize(xw))
        x_c = self.conv1(x_r)[:, : x_r.shape[1], :]
        x_c = self.ln_conv(x_c)
        x_res = x_r + x_c
        out, _ = self.lstm(x_res)
        return self.ln_lstm(out[:, -1, :])

    def __call__(self, xw: mx.array) -> mx.array:
        """(n_bars, seq, features) -> exposure (n_bars,) in [0, 1]."""
        z = self.encode(xw)
        w = mx.sigmoid(self.head(z)).squeeze(-1)
        if self.turnover_control != "band":
            return w
        b = nn.softplus(self.band(z)) * self.band_width
        lo = mx.clip(w - b[..., 0], 0.0, 1.0)
        hi = mx.clip(w + b[..., 1], 0.0, 1.0)
        return band_hold(w, lo, hi)
