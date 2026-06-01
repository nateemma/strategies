"""RealAutoencoder — per-class manifold reconstruction model.

One MLP autoencoder per NNNC class, trained ONLY on real same-class
samples to reconstruct them through a low-dimensional bottleneck. At
filter-time, the reconstruction MSE of each synth row measures how
far it sits from the real class-c manifold:

  * Low MSE → row sits on (or near) the learned manifold → keep.
  * High MSE → row is off-manifold (broken joints, missing feature
    structure, etc.) → drop.

Unlike Mahalanobis distance — which is symmetric around the centroid
and so rewards tightly-clustered samples — reconstruction error is
*manifold-aware*: a real tail sample reconstructs well as long as it
sits on the same manifold; a synth sample that lives near the centroid
but has broken joint structure reconstructs poorly. This is precisely
the failure mode (MODE_COLLAPSE with σ_ratio < 0.5) that Mahalanobis
amplifies.

Architecture: 24 (or whatever F is) → 64 → 32 → 16 → 32 → 64 → F.
GELU activations, no dropout (we want a tight fit so off-manifold
samples have high MSE). Linear output for clean reconstruction.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class RealAutoencoder(nn.Module):
    """Symmetric MLP autoencoder with a low-dimensional bottleneck."""

    def __init__(
        self,
        num_features: int,
        encoder_dims=(64, 32, 16),
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.encoder_dims = tuple(int(d) for d in encoder_dims)

        # Encoder: F → 64 → 32 → 16
        enc_layers = []
        in_dim = self.num_features
        for h in self.encoder_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.GELU())
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: 16 → 32 → 64 → F (mirror), final layer linear.
        dec_layers = []
        rev_dims = list(reversed(self.encoder_dims))[1:] + [self.num_features]
        in_dim = self.encoder_dims[-1]
        for i, h in enumerate(rev_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            if i < len(rev_dims) - 1:
                dec_layers.append(nn.GELU())
            in_dim = h
        self.decoder = nn.Sequential(*dec_layers)

    def __call__(self, x: mx.array) -> mx.array:
        """Encode-decode. Returns the reconstruction (same shape as x)."""
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: mx.array) -> mx.array:
        """Per-row MSE between x and its reconstruction. Returns (N,)."""
        recon = self(x)
        return mx.mean((x - recon) ** 2, axis=-1)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.save_weights(os.path.join(path, "autoencoder.safetensors"))
        meta = {
            "num_features": self.num_features,
            "encoder_dims": list(self.encoder_dims),
        }
        with open(os.path.join(path, "autoencoder_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> Optional["RealAutoencoder"]:
        weight_path = os.path.join(path, "autoencoder.safetensors")
        meta_path = os.path.join(path, "autoencoder_meta.pkl")
        if not (os.path.exists(weight_path) and os.path.exists(meta_path)):
            return None
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model = cls(
            num_features=int(meta["num_features"]),
            encoder_dims=tuple(meta.get("encoder_dims", (64, 32, 16))),
        )
        model.load_weights(weight_path)
        mx.eval(model.parameters())
        return model


def class_subdir(model_root: str, class_idx: int) -> str:
    """Conventional per-class save directory under a model root.
    Matches the realsignal classifier convention."""
    return os.path.join(model_root, f"class_{int(class_idx)}")
