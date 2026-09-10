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
