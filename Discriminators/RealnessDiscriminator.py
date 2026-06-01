"""RealnessDiscriminator — unified NN filter for GAN-synth quality.

Binary classifier that scores each candidate row by P(real) given its
features AND its claimed class label. Used by ``balance_single_task`` to
drop the bottom fraction of GAN-generated synth before the classifier
sees it.

Cross-label aware: the input concatenates feature vector + class one-hot,
so the discriminator learns class-conditional realness (catching e.g.
DDPM's Dirac artifact where class-2 synth has guard_metric_pos≠0 even
though real class-2 has it ≈ 0).

Trained ONCE on real samples (label 0) versus pooled synth from every
GAN family on disk (label 1, downsampled to match real count). The
unified-fakes training set teaches a generic "is this on the real
distribution" boundary that filters any GAN's output, not just the one
it was trained against.

Architecture: 4-layer MLP with GELU activations and dropout 0.2.
Output is a single logit; sigmoid is applied at scoring time.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class RealnessDiscriminator(nn.Module):
    """4-layer MLP discriminator: features + class one-hot → P(real).

    Hidden dims [512, 256, 128, 64] roughly match WGAN critic capacity at
    the input width but are deeper, since this discriminator is trained
    standalone (not in an adversarial loop) and benefits from more depth
    to capture joint structure that flagged broken in the augmentation
    diagnostics (joint |Δρ| up to 0.7 with sign flips on WGAN).
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dims=(512, 256, 128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout_p = float(dropout)

        in_dim = self.num_features + self.num_classes
        layers = []
        for h in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.dropout_p))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        """Forward pass. ``x``: (N, F). ``c``: (N, num_classes) one-hot.
        Returns (N,) logits — sigmoid at the call site if probabilities
        are needed."""
        xc = mx.concatenate([x, c], axis=-1)
        return self.net(xc).squeeze(-1)

    def score(self, x: mx.array, c: mx.array) -> mx.array:
        """Returns P(real) per row, in (0, 1). Higher = more real-looking."""
        # Discriminator is trained with label 0 = real, 1 = synth, so
        # sigmoid(logit) ≈ P(synth) and P(real) = 1 - that.
        return 1.0 - mx.sigmoid(self(x, c))

    def save(self, path: str) -> None:
        """Persist weights + architecture/scaler metadata to ``path``."""
        os.makedirs(path, exist_ok=True)
        self.save_weights(os.path.join(path, "realness_discriminator.safetensors"))
        meta = {
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout_p,
        }
        with open(os.path.join(path, "realness_discriminator_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> Optional["RealnessDiscriminator"]:
        """Restore a saved discriminator. Returns None if no model exists
        at ``path`` so callers can fall through cleanly."""
        weight_path = os.path.join(path, "realness_discriminator.safetensors")
        meta_path = os.path.join(path, "realness_discriminator_meta.pkl")
        if not (os.path.exists(weight_path) and os.path.exists(meta_path)):
            return None
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = cls(
            num_features=int(meta["num_features"]),
            num_classes=int(meta["num_classes"]),
            hidden_dims=tuple(meta.get("hidden_dims", (512, 256, 128, 64))),
            dropout=float(meta.get("dropout", 0.2)),
        )
        model.load_weights(weight_path)
        mx.eval(model.parameters())
        return model


def one_hot_class(class_indices: np.ndarray, num_classes: int) -> mx.array:
    """Build an (N, num_classes) one-hot matrix from class indices."""
    arr = np.asarray(class_indices).astype(int).flatten()
    oh = np.zeros((len(arr), num_classes), dtype=np.float32)
    oh[np.arange(len(arr)), arr] = 1.0
    return mx.array(oh)
