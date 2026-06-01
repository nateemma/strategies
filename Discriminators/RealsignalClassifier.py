"""RealsignalClassifier — per-row binary signal classifier.

One model per NNNC class. Each model answers a single question: "given
just this row's features, does this look like a real example of class c
or not?" Trained ONLY on real data — y=1 for rows the labeler marked as
class c, y=0 for every other class. No GAN samples involved at training
or inference time.

At filter-time, a candidate synth row claiming class c is scored by
classifier c. Low score means "doesn't look like a real class c" → drop.

The "no GAN" property is the whole point — the score reflects the real
signal pattern the NNNC classifier itself is trying to learn, not any
particular synth-generation artifact.

Architecture: 4-layer MLP, GELU activations, dropout 0.2. Single sigmoid
output. Identical capacity to RealnessDiscriminator minus the class
one-hot input (since each model handles one class).
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class RealsignalClassifier(nn.Module):
    """4-layer MLP per-row binary classifier: features → P(real class c).
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims=(512, 256, 128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout_p = float(dropout)

        in_dim = self.num_features
        layers = []
        for h in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.dropout_p))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. ``x``: (N, F). Returns (N,) logits."""
        return self.net(x).squeeze(-1)

    def score(self, x: mx.array) -> mx.array:
        """Returns P(real class c) per row, in (0, 1). Higher = more
        signal-like."""
        return mx.sigmoid(self(x))

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.save_weights(os.path.join(path, "realsignal.safetensors"))
        meta = {
            "num_features": self.num_features,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout_p,
        }
        with open(os.path.join(path, "realsignal_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> Optional["RealsignalClassifier"]:
        weight_path = os.path.join(path, "realsignal.safetensors")
        meta_path = os.path.join(path, "realsignal_meta.pkl")
        if not (os.path.exists(weight_path) and os.path.exists(meta_path)):
            return None
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = cls(
            num_features=int(meta["num_features"]),
            hidden_dims=tuple(meta.get("hidden_dims", (512, 256, 128, 64))),
            dropout=float(meta.get("dropout", 0.2)),
        )
        model.load_weights(weight_path)
        mx.eval(model.parameters())
        return model


def class_subdir(model_root: str, class_idx: int) -> str:
    """Conventional per-class save directory under a model root."""
    return os.path.join(model_root, f"class_{int(class_idx)}")
