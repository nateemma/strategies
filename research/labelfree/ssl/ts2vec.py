"""TS2Vec in MLX (T011-T014).

Yue et al. 2022: a dilated CAUSAL CNN encoder trained by hierarchical contrastive loss, with no
labels anywhere.

CAUSAL BY CONSTRUCTION, which is what makes FR-002 satisfiable rather than merely checkable.
Every convolution is left-padded so output t sees only inputs <= t. The deletion test in
causality.py then MEASURES that rather than trusting it — the architecture is an argument, and
this study's rule is that an argument is not evidence.

HYPERPARAMETERS ARE TS2VEC DEFAULTS AND ARE NOT SEARCHED. Searching them across four screens
would be the multiple-testing failure brief 05 measured at PBO 0.7128: with enough configurations
something clears any bar.

MLX, and the one pitfall this repo has already paid for: EVERY UPDATED ARRAY must appear in the
per-step `mx.eval`. Omitting one lets the graph grow until Metal's ~500K limit trips.
"""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

WIDTH = 320            # TS2Vec default output width
HIDDEN = 64
DEPTH = 10             # dilations 1,2,4,...,512
CROP_MIN = 64
EPOCHS = 20
BATCH = 16
LR = 1e-3


class DilatedBlock(nn.Module):
    """One residual dilated CAUSAL conv block."""

    def __init__(self, cin: int, cout: int, dilation: int, k: int = 3):
        super().__init__()
        self.pad = (k - 1) * dilation          # LEFT pad only — this is the causality
        self.conv1 = nn.Conv1d(cin, cout, k, dilation=dilation)
        self.conv2 = nn.Conv1d(cout, cout, k, dilation=dilation)
        self.proj = nn.Conv1d(cin, cout, 1) if cin != cout else None

    def __call__(self, x):                     # x: (B, T, C)
        r = x if self.proj is None else self.proj(x)
        h = mx.pad(x, [(0, 0), (self.pad, 0), (0, 0)])
        h = nn.gelu(self.conv1(h))
        h = mx.pad(h, [(0, 0), (self.pad, 0), (0, 0)])
        h = self.conv2(h)
        return h + r


class TS2Vec(nn.Module):
    def __init__(self, n_in: int, width: int = WIDTH, hidden: int = HIDDEN, depth: int = DEPTH):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.blocks = [DilatedBlock(hidden, hidden, 2 ** i) for i in range(depth)]
        self.out = nn.Conv1d(hidden, width, 1)

    def __call__(self, x):                     # (B, T, n_in) -> (B, T, width)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out(h)


def _hierarchical_contrastive(z1, z2, temp: float = 1.0):
    """Temporal + instance-wise contrastive loss at successive pooling levels."""
    loss, d = mx.array(0.0), 0
    while z1.shape[1] > 1:
        loss = loss + _inst_loss(z1, z2, temp) + _temp_loss(z1, z2, temp)
        d += 1
        n = (z1.shape[1] // 2) * 2
        z1 = mx.mean(z1[:, :n].reshape(z1.shape[0], -1, 2, z1.shape[2]), axis=2)
        z2 = mx.mean(z2[:, :n].reshape(z2.shape[0], -1, 2, z2.shape[2]), axis=2)
        if d >= 6:
            break
    return loss / max(1, d)


def _inst_loss(z1, z2, temp):
    """Same timestamp, different instances: the matching instance is the positive."""
    B = z1.shape[0]
    if B < 2:
        return mx.array(0.0)
    a = mx.transpose(z1, [1, 0, 2])                       # (T, B, W)
    b = mx.transpose(z2, [1, 0, 2])
    sim = (a @ mx.transpose(b, [0, 2, 1])) / temp         # (T, B, B)
    lab = mx.arange(B)
    return mx.mean(nn.losses.cross_entropy(sim.reshape(-1, B), mx.tile(lab, [sim.shape[0]])))


def _temp_loss(z1, z2, temp):
    """Same instance, different timestamps: the matching timestamp is the positive."""
    T = z1.shape[1]
    if T < 2:
        return mx.array(0.0)
    sim = (z1 @ mx.transpose(z2, [0, 2, 1])) / temp       # (B, T, T)
    lab = mx.arange(T)
    return mx.mean(nn.losses.cross_entropy(sim.reshape(-1, T), mx.tile(lab, [sim.shape[0]])))


def fit(X: np.ndarray, seed: int | None = None, epochs: int = EPOCHS,
        batch: int = BATCH, width: int = WIDTH, verbose: bool = False) -> TS2Vec:
    """Train on ONE split's bars, without labels. X: (T, n_in)."""
    seed = int(os.environ.get("MLX_SEED", 7)) if seed is None else seed
    mx.random.seed(seed)
    np.random.seed(seed)

    model = TS2Vec(X.shape[1], width=width)
    opt = optim.Adam(learning_rate=LR)
    Xm = mx.array(np.ascontiguousarray(X, dtype=np.float32))
    T = Xm.shape[0]
    win = min(256, max(CROP_MIN * 2, T // 8))
    if T < win + 2:
        raise ValueError(f"series too short for TS2Vec: {T} bars, need > {win + 2}")

    def loss_fn(m, a, b):
        return _hierarchical_contrastive(m(a), m(b))

    gfn = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(seed)
    losses = []
    for ep in range(epochs):
        starts = rng.integers(0, T - win, size=batch)
        crops = mx.stack([Xm[s:s + win] for s in starts])
        o1, o2 = CROP_MIN // 2, 0
        a, b = crops[:, o1:], crops[:, o2:win - o1]
        loss, grads = gfn(model, a, b)
        opt.update(model, grads)
        # FR-005: EVERY updated array in the per-step eval — model params AND optimiser state.
        # Omitting one lets the graph grow until Metal's ~500K limit trips.
        mx.eval(model.parameters(), opt.state, loss)
        losses.append(float(loss))
        if verbose and ep % 5 == 0:
            print(f"    epoch {ep:>3}  loss {float(loss):.4f}", flush=True)
    model._losses = losses
    return model


def embed(model: TS2Vec, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Per-bar embeddings for the whole series, in causal chunks."""
    out = []
    for i in range(0, len(X), chunk):
        seg = mx.array(np.ascontiguousarray(X[:i + chunk], dtype=np.float32))[None]
        z = model(seg)[0]
        mx.eval(z)
        out.append(np.array(z)[i:i + chunk])
    return np.concatenate(out, axis=0)
