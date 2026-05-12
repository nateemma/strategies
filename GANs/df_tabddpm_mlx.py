"""
TabDDPMMLX — MLX-native, continuous-only, single-task TabDDPM trainer
and sampler.

Lifecycle: fit() / generate() / save() / load() — same shape as WGANMLX
so the GANInterface backend adapter is a thin wrapper.

The diffusion math lives in `diffusion_mlx`; this module owns the model
class (MLP backbone + time embedding + class embedding), the training
loop, the EMA copy used for sampling, and the safetensors + pickle
save/load lifecycle.

See `docs/superpowers/specs/2026-05-11-tabddpm-design.md` for the
design and `docs/superpowers/plans/2026-05-11-tabddpm-implementation.md`
for the implementation plan.
"""

from __future__ import annotations

import math
import os
import pickle
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from GANs.diffusion_mlx import Schedule, ddim_sample, make_schedule, q_sample


def _tree_copy(params: Any) -> Any:
    """Deep-copy a nested mlx parameter tree (dict / list / mx.array).

    Used to snapshot best-loss EMA params during fit() so we can
    restore them at the end if late-epoch training degrades."""
    if isinstance(params, dict):
        return {k: _tree_copy(v) for k, v in params.items()}
    if isinstance(params, list):
        return [_tree_copy(p) for p in params]
    if isinstance(params, mx.array):
        return mx.array(params)
    return params


def _tree_lerp(a: Any, b: Any, t: float) -> Any:
    """Element-wise (1-t)*a + t*b on nested mlx parameter trees.

    Used for EMA updates — a is the EMA params, b is the live params,
    t = 1 - ema_decay (so output stays closer to a when decay is high).
    """
    if isinstance(a, dict):
        return {k: _tree_lerp(a[k], b[k], t) for k in a}
    if isinstance(a, list):
        return [_tree_lerp(ai, bi, t) for ai, bi in zip(a, b)]
    if isinstance(a, mx.array):
        return (1.0 - t) * a + t * b
    return a  # non-array leaves passed through


_META_FILENAME = "tabddpm_metadata.pkl"
_WEIGHTS_FILENAME = "tabddpm_gen_mlx.safetensors"


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class _SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps,
    followed by two SiLU-activated Linear layers projecting to d_model.
    Same shape the TabDDPM paper uses."""

    def __init__(self, d_model: int, sinusoid_dim: int = 128):
        super().__init__()
        self.sinusoid_dim = sinusoid_dim
        self.proj1 = nn.Linear(sinusoid_dim, d_model)
        self.proj2 = nn.Linear(d_model, d_model)

    def __call__(self, t: mx.array) -> mx.array:
        # t: (B,) int32 → (B, sinusoid_dim) sin/cos features → (B, d_model).
        half = self.sinusoid_dim // 2
        freqs = mx.exp(
            -math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half
        )
        args = t.astype(mx.float32)[:, None] * freqs[None, :]
        emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
        emb = nn.silu(self.proj1(emb))
        emb = nn.silu(self.proj2(emb))
        return emb


class _MLPBlock(nn.Module):
    """Linear → ReLU → Dropout, the TabDDPM paper's block primitive."""

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.linear(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class _TabDDPMMLP(nn.Module):
    """MLP backbone: x_proj + t_embed + class_embed → stacked blocks → head."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 256,
        d_layers: Sequence[int] = (256, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.x_proj = nn.Linear(num_features, d_model)
        self.t_embed = _SinusoidalTimeEmbed(d_model)
        # +1 slot for the "null" / unconditional class token used by
        # classifier-free guidance. fit() replaces real class indices
        # with num_classes (the null slot) with probability p_uncond,
        # training the model to predict both conditional and
        # unconditional ε. generate() blends the two at sample time
        # when guidance_scale != 1.0.
        self.class_embed = nn.Embedding(num_classes + 1, d_model)

        dims = [d_model, *d_layers]
        self.blocks = [
            _MLPBlock(dims[i], dims[i + 1], dropout=dropout)
            for i in range(len(dims) - 1)
        ]
        self.head = nn.Linear(dims[-1], num_features)

    def __call__(self, x_t: mx.array, t: mx.array, class_idx: mx.array) -> mx.array:
        h = self.x_proj(x_t) + self.t_embed(t) + self.class_embed(class_idx)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# TabDDPMMLX — outer class
# ---------------------------------------------------------------------------


class TabDDPMMLX:
    """Continuous-only, single-task, MLX-native TabDDPM.

    Construct with feature/class dimensions and (optional) hyperparams;
    call fit(data, labels) once; then generate() / save() / load().
    """

    def __init__(
        self,
        num_features: int = 0,
        num_classes: int = 0,
        *,
        d_model: int = 256,
        d_layers: Sequence[int] = (256, 256),
        dropout: float = 0.0,
        num_timesteps: int = 1000,
        num_sample_steps: int = 50,
        epochs: int = 300,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        ema_decay: float = 0.999,
        eval_frequency: int = 20,
        lr_min_ratio: float = 0.01,
        min_snr_gamma: float = 5.0,
        p_uncond: float = 0.1,
        guidance_scale: float = 1.0,
        verbose: bool = True,
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.d_model = d_model
        self.d_layers = tuple(d_layers)
        self.dropout = dropout
        self.num_timesteps = num_timesteps
        self.num_sample_steps = num_sample_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.eval_frequency = eval_frequency
        self.lr_min_ratio = lr_min_ratio
        self.min_snr_gamma = min_snr_gamma
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale
        self.verbose = verbose

        # Feature stats populated by fit(); used by _postprocess.
        self.feature_min: Optional[np.ndarray] = None
        self.feature_max: Optional[np.ndarray] = None

        # Models created lazily in fit() once we know num_features/num_classes.
        # Skeleton instantiation (e.g. before load_from) still needs the
        # MLPs so the test can inspect their shapes — only build them when
        # dimensions are known.
        if num_features > 0 and num_classes > 0:
            self._build_models()
        else:
            self._mlp = None
            self._ema_mlp = None

        self._sched: Schedule = make_schedule(self.num_timesteps)

    def _build_models(self) -> None:
        self._mlp = _TabDDPMMLP(
            self.num_features, self.num_classes,
            d_model=self.d_model, d_layers=self.d_layers,
            dropout=self.dropout,
        )
        self._ema_mlp = _TabDDPMMLP(
            self.num_features, self.num_classes,
            d_model=self.d_model, d_layers=self.d_layers,
            dropout=self.dropout,
        )

    # ---------- training ---------- #

    def _minmax_fit(self, data: np.ndarray) -> np.ndarray:
        """Compute per-column min/max, scale data to [-1, 1].

        Stores stats on self for use in _postprocess; returns the scaled array.
        """
        self.feature_min = data.min(axis=0).astype(np.float32)
        self.feature_max = data.max(axis=0).astype(np.float32)
        rng = self.feature_max - self.feature_min
        rng = np.where(rng == 0, 1.0, rng)
        return ((data - self.feature_min) / rng * 2.0 - 1.0).astype(np.float32)

    def _minmax_invert(self, x: np.ndarray) -> np.ndarray:
        rng = self.feature_max - self.feature_min
        rng = np.where(rng == 0, 1.0, rng)
        return ((x + 1.0) / 2.0) * rng + self.feature_min

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        categorical_columns: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> None:
        """Train the diffusion model.

        Args:
            data:                (N, F) float32 — continuous features only.
            labels:              (N, C) one-hot float32.
            categorical_columns: Warned about and dropped (the v1 MLX
                                 TabDDPM is continuous-only — same policy
                                 as MLX CTAB-GAN).
        """
        if categorical_columns:
            print(
                f"[TabDDPMMLX] categorical_columns={list(categorical_columns)} "
                "ignored — this backend is continuous-only."
            )

        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"data must be 2-D (N, F); got shape {data.shape}")
        if labels.ndim != 2:
            raise ValueError(f"labels must be 2-D (N, C); got shape {labels.shape}")

        # Lazy-init the model now that we know dimensions.
        if self.num_features == 0:
            self.num_features = data.shape[1]
        if self.num_classes == 0:
            self.num_classes = labels.shape[1]
        if self._mlp is None:
            self._build_models()

        data_norm = self._minmax_fit(data)
        class_idx_np = labels.argmax(axis=1).astype(np.int32)
        N = data_norm.shape[0]

        # Move to MLX arrays once.
        data_mx = mx.array(data_norm)
        class_idx_mx = mx.array(class_idx_np)

        optimizer = optim.AdamW(learning_rate=self.learning_rate,
                                weight_decay=self.weight_decay)

        # Class-balanced batch-sampling weights.  The GAN's purpose is
        # generating minority-class samples, but with uniform sampling
        # the model sees minority-class conditioning rarely during
        # training — exactly the conditioning path that matters most at
        # sample time.  Reweight per-sample probabilities so each class
        # contributes equal total probability; within a class the
        # samples remain uniform.
        class_counts = np.bincount(
            class_idx_np, minlength=self.num_classes
        ).astype(np.float32)
        class_weights = np.where(
            class_counts > 0, 1.0 / np.maximum(class_counts, 1.0), 0.0
        )
        per_sample_weights = class_weights[class_idx_np]
        wsum = per_sample_weights.sum()
        if wsum > 0:
            per_sample_weights = per_sample_weights / wsum
        else:
            per_sample_weights = None  # degenerate: no classes seen

        def loss_fn(model: _TabDDPMMLP, x0: mx.array, t: mx.array,
                    noise: mx.array, cls: mx.array) -> mx.array:
            x_t = q_sample(x0, t, noise, self._sched)
            eps_hat = model(x_t, t, cls)
            # Per-sample squared error, mean over feature dim → (B,).
            per_sample_sq_err = mx.mean((eps_hat - noise) ** 2, axis=-1)
            # Min-SNR-γ weighting (Hang et al. 2023). For ε-parameterised
            # diffusion, w_t = min(SNR_t, γ) / SNR_t. Down-weights easy
            # (high-SNR) timesteps so the model spends more capacity on
            # the hard middle range of the noise schedule. ~5-10% sample
            # quality gain on standard benchmarks.
            ac_t = self._sched.alphas_cumprod[t]
            one_minus_ac = mx.maximum(1.0 - ac_t, 1e-8)
            snr_t = ac_t / one_minus_ac
            weights = mx.minimum(snr_t, self.min_snr_gamma) / snr_t
            return mx.mean(weights * per_sample_sq_err)

        loss_and_grad = nn.value_and_grad(self._mlp, loss_fn)

        # Initialise EMA params to live params.
        self._ema_mlp.update(self._mlp.parameters())

        steps_per_epoch = max(1, N // self.batch_size)
        total_steps = max(1, self.epochs * steps_per_epoch)
        rng = np.random.default_rng(0)

        # Best-EMA snapshot (by training loss). Mirrors WGAN-MLX's
        # "save best, restore at end if final is worse" pattern — no
        # val split, just per-epoch training loss as the signal.
        best_loss = float("inf")
        best_ema_params: Optional[Any] = None
        global_step = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for _ in range(steps_per_epoch):
                # Cosine LR decay from self.learning_rate to
                # learning_rate * lr_min_ratio over total_steps.
                optimizer.learning_rate = self._cosine_lr(global_step, total_steps)
                global_step += 1

                if per_sample_weights is not None:
                    idx = rng.choice(
                        N, size=self.batch_size, replace=True, p=per_sample_weights,
                    )
                else:
                    idx = rng.integers(0, N, size=self.batch_size)
                idx_mx = mx.array(idx, dtype=mx.int32)
                x0 = data_mx[idx_mx]
                cls = class_idx_mx[idx_mx]

                # Classifier-free guidance: drop the class condition with
                # probability p_uncond, replacing it with the null token
                # (index num_classes).  The model thereby learns both the
                # conditional and unconditional noise prediction, and
                # generate() can blend them at sample time.
                if self.p_uncond > 0.0:
                    drop = rng.random(self.batch_size) < self.p_uncond
                    if drop.any():
                        cls_np = np.asarray(cls)
                        cls_np = np.where(drop, self.num_classes, cls_np).astype(np.int32)
                        cls = mx.array(cls_np)

                t = mx.random.randint(0, self.num_timesteps, (self.batch_size,))
                noise = mx.random.normal((self.batch_size, self.num_features))

                loss, grads = loss_and_grad(self._mlp, x0, t, noise, cls)
                optimizer.update(self._mlp, grads)

                # EMA update: θ_ema ← decay·θ_ema + (1-decay)·θ
                self._ema_update()

                # Force materialisation of ALL live state on every step.
                # Without the EMA params in this eval list, MLX's lazy
                # graph accumulates _ema_mlp tensors across steps and
                # eventually trips Metal's per-process resource limit
                # (~500K allocations) — confirmed crash at epoch ~63
                # on a 720-day training run.
                mx.eval(
                    self._mlp.parameters(),
                    self._ema_mlp.parameters(),
                    optimizer.state,
                    loss,
                )
                epoch_loss += float(loss.item())

            avg = epoch_loss / steps_per_epoch

            # Best-EMA snapshot — track every epoch so we can restore the
            # best version if late-epoch training degrades.
            marker = ""
            if avg < best_loss:
                best_loss = avg
                best_ema_params = _tree_copy(self._ema_mlp.parameters())
                marker = " *"

            if self.verbose:
                lr_now = float(optimizer.learning_rate)
                print(
                    f"[TabDDPMMLX] epoch {epoch+1}/{self.epochs}  "
                    f"loss={avg:.4f}  lr={lr_now:.6f}  best={best_loss:.4f}{marker}"
                )

        # Restore best EMA snapshot if we ever took one (always true
        # when we ran at least one epoch).
        if best_ema_params is not None:
            self._ema_mlp.update(best_ema_params)
            if self.verbose:
                print(
                    f"[TabDDPMMLX] restored best EMA params "
                    f"(best loss={best_loss:.4f})"
                )

    def _ema_update(self) -> None:
        decay = self.ema_decay
        live = self._mlp.parameters()
        ema = self._ema_mlp.parameters()
        new_ema = _tree_lerp(ema, live, 1.0 - decay)
        self._ema_mlp.update(new_ema)

    def _cosine_lr(self, step: int, total_steps: int) -> float:
        """Cosine LR decay from self.learning_rate down to
        learning_rate * lr_min_ratio over [0, total_steps].

        Past total_steps, returns the floor — guards against any
        last-step overrun where global_step == total_steps.
        """
        if total_steps <= 0:
            return self.learning_rate
        lr_max = float(self.learning_rate)
        lr_min = lr_max * float(self.lr_min_ratio)
        progress = min(float(step) / float(total_steps), 1.0)
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))

    # ---------- sampling ---------- #

    def generate(self, n: int, one_hot: np.ndarray) -> np.ndarray:
        """Sample n synthetic rows conditioned on `one_hot`.

        Args:
            n:       Number of samples.
            one_hot: (n, num_classes) float32.

        Returns:
            (n, 1, num_features) float32 numpy array. The trailing seq
            axis exists so `balance_single_task`'s _SQUEEZE_SEQ_DIM_TYPES
            path can squeeze it — matches the WGAN convention.
        """
        if self._ema_mlp is None:
            raise RuntimeError("TabDDPMMLX.generate called before fit/load.")

        one_hot = np.asarray(one_hot, dtype=np.float32)
        if one_hot.shape != (n, self.num_classes):
            raise ValueError(
                f"one_hot must be ({n}, {self.num_classes}); got {one_hot.shape}"
            )
        class_idx = mx.array(one_hot.argmax(axis=1).astype(np.int32))

        # Closure over the EMA model so the diffusion module stays
        # model-agnostic. eval() disables dropout for sampling — matters
        # when callers set dropout>0 at training time.
        ema = self._ema_mlp
        ema.eval()

        # Classifier-free guidance: when guidance_scale != 1.0, blend
        # the conditional and unconditional ε predictions per step:
        #   ε̂ = ε̂_u + w · (ε̂_c − ε̂_u)
        # scale=1 is plain conditional sampling (one model call/step);
        # scale>1 amplifies the conditional direction (sharper class
        # adherence at the cost of one extra forward pass).
        guidance = float(self.guidance_scale)
        if guidance == 1.0:
            def model_fn(x_t: mx.array, t: mx.array, cond: mx.array) -> mx.array:
                return ema(x_t, t, cond)
        else:
            null_idx = mx.full((n,), self.num_classes, dtype=mx.int32)

            def model_fn(x_t: mx.array, t: mx.array, cond: mx.array) -> mx.array:
                eps_cond = ema(x_t, t, cond)
                eps_uncond = ema(x_t, t, null_idx)
                return eps_uncond + guidance * (eps_cond - eps_uncond)

        try:
            x0_mx = ddim_sample(
                model_fn=model_fn,
                shape=(n, self.num_features),
                cond=class_idx,
                sched=self._sched,
                num_steps=self.num_sample_steps,
            )
        finally:
            ema.train()

        # _postprocess: clip to [-1, 1], then inverse minmax.
        x0_np = np.clip(np.asarray(x0_mx), -1.0, 1.0)
        x0_np = self._minmax_invert(x0_np)
        return x0_np.reshape(n, 1, self.num_features).astype(np.float32)

    # ---------- persistence ---------- #

    def save(self, save_path: str, **extra_metadata: Any) -> None:
        """Persist the EMA model + ctor params + feature stats.

        extra_metadata (e.g. MASTER_MIN_BUY_GAIN_THRESHOLD) is merged
        into the pickle so GANInterface.load(expected=...) can validate
        thresholds at load time.
        """
        if self._ema_mlp is None:
            raise RuntimeError("TabDDPMMLX.save called before fit.")
        os.makedirs(save_path, exist_ok=True)

        self._ema_mlp.save_weights(os.path.join(save_path, _WEIGHTS_FILENAME))

        meta: Dict[str, Any] = {
            "num_features":     self.num_features,
            "num_classes":      self.num_classes,
            "d_model":          self.d_model,
            "d_layers":         list(self.d_layers),
            "dropout":          self.dropout,
            "num_timesteps":    self.num_timesteps,
            "num_sample_steps": self.num_sample_steps,
            "p_uncond":         self.p_uncond,
            "guidance_scale":   self.guidance_scale,
            "feature_min":      np.asarray(self.feature_min, dtype=np.float32),
            "feature_max":      np.asarray(self.feature_max, dtype=np.float32),
        }
        meta.update(extra_metadata)
        with open(os.path.join(save_path, _META_FILENAME), "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load_from(cls, save_path: str) -> Tuple["TabDDPMMLX", Dict[str, Any]]:
        meta_p = os.path.join(save_path, _META_FILENAME)
        weights_p = os.path.join(save_path, _WEIGHTS_FILENAME)
        if not (os.path.exists(meta_p) and os.path.exists(weights_p)):
            raise FileNotFoundError(
                f"No MLX-format TabDDPM model at {save_path} "
                f"(needs {_META_FILENAME} + {_WEIGHTS_FILENAME})"
            )

        with open(meta_p, "rb") as f:
            metadata = pickle.load(f)

        instance = cls(
            num_features=int(metadata["num_features"]),
            num_classes=int(metadata["num_classes"]),
            d_model=int(metadata.get("d_model", 256)),
            d_layers=tuple(metadata.get("d_layers", (256, 256))),
            dropout=float(metadata.get("dropout", 0.0)),
            num_timesteps=int(metadata.get("num_timesteps", 1000)),
            num_sample_steps=int(metadata.get("num_sample_steps", 50)),
            p_uncond=float(metadata.get("p_uncond", 0.1)),
            guidance_scale=float(metadata.get("guidance_scale", 1.0)),
            verbose=False,
        )
        instance._ema_mlp.load_weights(weights_p)
        instance.feature_min = np.asarray(metadata["feature_min"], dtype=np.float32)
        instance.feature_max = np.asarray(metadata["feature_max"], dtype=np.float32)
        return instance, metadata
