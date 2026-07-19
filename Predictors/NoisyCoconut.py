"""
NoisyCoconut — training-free latent multi-path prediction for MLX classifiers.

Wraps a trained classifier's ``predict()`` with the NoisyCoconut mechanism:
perturb a continuous representation K times to spawn diverging "reasoning
paths", decode each, and aggregate by probability-mass voting (mean-softmax).
No retraining — this reuses the already-trained model.

Two perturbation spaces (mixin attribute ``noisy_perturb_space``):
  - "input"  : perturb the (already-normalised) input tensor, run the full
               model K times. Works with any backbone — the Stage-0 pre-gate.
  - "latent" : perturb the model's latent (``encode`` output) and run only the
               cheap ``decode`` head K times. Requires the backbone to expose
               ``encode``/``decode`` (e.g. the split ``_LSTMModel``). This is
               the actual COCONUT mechanism.

Determinism: ``predict()`` is batch-called once over the dataframe in backtest,
so seeding the MLX RNG once at the top makes identical backtests reproduce
exactly. ``noisy_sigma = 0`` returns exactly the production softmax (identity),
which doubles as a correctness check.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


class NoisyCoconutMixin:
    """Override ``predict()`` with NoisyCoconut latent/input multi-path voting.

    Mix in *before* the concrete MLX classifier so this ``predict`` wins the MRO,
    e.g. ``class Foo(NoisyCoconutMixin, NNNClassifierMLX_LSTM): ...``.
    """

    # Perturbation space: "latent" (COCONUT) or "input" (Stage-0 pre-gate).
    noisy_perturb_space: str = "latent"
    noisy_sigma: float = 0.1
    noisy_k: int = 16
    noisy_seed: int = 42

    def predict(self, data) -> np.ndarray:
        # lazy load (identical to MLXClassifierNary.predict)
        if self.model is None:
            self.model = self.load()
        if self.model is None:
            raise RuntimeError(
                f"CRITICAL: No MLX model found for {self.name} at {self.model_path}. "
                "Ensure training completed successfully."
            )

        # accept DataFrame or numpy tensor (identical to base predict)
        if self.dataframeUtils.is_dataframe(data):
            tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len, method=3)
        else:
            tensor = np.array(data)

        self.model.eval()
        X = mx.array(tensor, dtype=mx.float32)

        k = int(self.noisy_k)
        sigma = float(self.noisy_sigma)
        # Seed once → reproducible backtests. sigma=0 makes every path identical
        # to the unperturbed forward, so the mean equals the production softmax.
        mx.random.seed(int(self.noisy_seed))

        if self.noisy_perturb_space == "latent":
            if not (hasattr(self.model, "encode") and hasattr(self.model, "decode")):
                raise RuntimeError(
                    f"latent perturbation requires encode()/decode() on "
                    f"{type(self.model).__name__}; use noisy_perturb_space='input'."
                )
            h = self.model.encode(X)  # (B, F) — computed once
            mx.eval(h)
            # Scale noise per latent dimension by its batch std so every dim is
            # perturbed on its own scale.
            std = mx.std(h, axis=0, keepdims=True)  # (1, F)
            acc = None
            for _ in range(k):
                h_k = h + sigma * std * mx.random.normal(h.shape)
                p_k = self.model.decode(h_k)
                acc = p_k if acc is None else acc + p_k
            preds = acc / k

        elif self.noisy_perturb_space == "input":
            acc = None
            for _ in range(k):
                x_k = X + sigma * mx.random.normal(X.shape)
                p_k = self.model(x_k)
                acc = p_k if acc is None else acc + p_k
            preds = acc / k

        else:
            raise ValueError(
                f"noisy_perturb_space must be 'latent' or 'input', "
                f"got {self.noisy_perturb_space!r}"
            )

        mx.eval(preds)
        return np.array(preds)
