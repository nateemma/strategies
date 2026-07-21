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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from GANs.diffusion_mlx import Schedule, ddim_sample, make_schedule, q_sample
from GANs.diffusion_edm_mlx import (
    DEFAULT_P_MEAN, DEFAULT_P_STD,
    DEFAULT_SIGMA_MIN, DEFAULT_SIGMA_MAX, DEFAULT_RHO,
    DEFAULT_SIGMA_DATA,
    build_sigma_schedule, heun_sample, sample_log_normal_sigma,
)


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
    """Pre-LN + SiLU + Linear with residual connection.

    Modern DDPM block pattern (post-2021 TabDDPM follow-ups, improved
    DDPM works) — LayerNorm before the linear stabilizes gradient flow
    at depth, SiLU is smoother than ReLU (helps diffusion-noise
    prediction), and the residual lets information bypass the block
    when the linear's contribution is small. Replaces the prior
    Linear→ReLU→Dropout primitive.

    When ``d_in != d_out`` the residual path projects via a 1×1 linear
    so the addition is dimension-correct. With current defaults
    (d_model=256, d_layers=(256, 256)) all blocks are square and the
    projection is the identity, but we keep the branch for safety so
    subclasses can grow the hidden width without breaking the residual.
    """

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.linear = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else None

    def __call__(self, x: mx.array) -> mx.array:
        residual = x if self.proj is None else self.proj(x)
        h = self.linear(nn.silu(self.norm(x)))
        if self.dropout is not None:
            h = self.dropout(h)
        return residual + h


class _FeatureAttentionBlock(nn.Module):
    """Per-feature self-attention block.

    Each of ``num_features`` scalar inputs is mapped to a ``d_feature``
    embedding (shared value→vector projection + per-feature learnable ID
    embedding), then multi-head self-attention runs across the feature
    tokens. The attended tokens are flattened and projected to ``d_model``
    so the block's output can be added to ``x_proj(x)`` as a residual.

    Why this exists: the shared-weight MLP blocks can model joint
    feature interactions only implicitly via mixing weights. With 25
    features and many subtle pairwise correlations (rsi × log_volume,
    bb_width × adx, etc.), the MLP under-fits those joint structures
    even at depth. Attention treats features as tokens and learns
    explicit pairwise weights → directly addresses the inter-indicator
    Δρ that's been stuck in the diagnostic across earlier improvements.
    """

    def __init__(
        self,
        num_features: int,
        d_feature: int = 32,
        n_heads: int = 4,
        d_model: int = 256,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_feature = d_feature
        # Shared scalar → d_feature projection. Each feature value gets
        # the same mapping; the per-feature distinction comes from the
        # ID embedding below.
        self.value_proj = nn.Linear(1, d_feature)
        # Per-feature ID embedding. Gives feature i a unique "address"
        # vector so attention can specialize per (feature_i, feature_j)
        # pair instead of treating tokens interchangeably.
        self.feat_id_embed = nn.Embedding(num_features, d_feature)
        self.norm = nn.LayerNorm(d_feature)
        self.attn = nn.MultiHeadAttention(dims=d_feature, num_heads=n_heads)
        # Aggregate to d_model via flatten + linear so the output can be
        # summed with the main x_proj path.
        self.out_proj = nn.Linear(num_features * d_feature, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, num_features)
        batch = x.shape[0]
        value_emb = self.value_proj(mx.expand_dims(x, axis=-1))  # (B, F, d_f)
        ids = mx.arange(self.num_features)
        id_emb = self.feat_id_embed(ids)  # (F, d_f)
        tokens = value_emb + mx.expand_dims(id_emb, axis=0)  # (B, F, d_f)
        # Pre-LN + self-attention + residual within the token space.
        normed = self.norm(tokens)
        attended = self.attn(normed, normed, normed)
        out_tokens = tokens + attended
        flat = out_tokens.reshape(batch, self.num_features * self.d_feature)
        return self.out_proj(flat)


class _TabDDPMMLP(nn.Module):
    """MLP backbone: x_proj + t_embed + class_embed (+ pair_embed) → blocks → head.

    Pair conditioning is optional and gated on ``num_pairs > 0``: when set,
    a parallel ``pair_embed`` is added to the same hidden state as the
    class embedding, letting the model produce per-pair-conditional
    samples (mirrors the MLX CTAB-GAN pattern).
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 256,
        d_layers: Sequence[int] = (256, 256),
        dropout: float = 0.0,
        num_pairs: int = 0,
        use_attention: bool = True,
        attn_d_feature: int = 32,
        attn_n_heads: int = 4,
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
        self.num_pairs = num_pairs
        if num_pairs > 0:
            self.pair_embed = nn.Embedding(num_pairs, d_model)
        else:
            self.pair_embed = None
        # Optional cross-feature self-attention head. Added as a residual
        # to the main x_proj path, so use_attention=False degrades the
        # model to the pre-Step-3 architecture exactly. Step 3 of the
        # GAN improvement plan: explicit pairwise feature modelling.
        if use_attention:
            self.feat_attn = _FeatureAttentionBlock(
                num_features=num_features,
                d_feature=attn_d_feature,
                n_heads=attn_n_heads,
                d_model=d_model,
            )
        else:
            self.feat_attn = None

        dims = [d_model, *d_layers]
        self.blocks = [
            _MLPBlock(dims[i], dims[i + 1], dropout=dropout)
            for i in range(len(dims) - 1)
        ]
        self.head = nn.Linear(dims[-1], num_features)

    def __call__(
        self,
        x_t: mx.array,
        t: mx.array,
        class_idx: mx.array,
        pair_idx: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.x_proj(x_t) + self.t_embed(t) + self.class_embed(class_idx)
        if self.pair_embed is not None and pair_idx is not None:
            h = h + self.pair_embed(pair_idx)
        if self.feat_attn is not None:
            h = h + self.feat_attn(x_t)
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
        num_pairs: int = 0,
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
        # min_snr_gamma > 0 enables Min-SNR-γ loss weighting (Hang et
        # al. 2023). Default 0.0 — uniform timestep weighting, matching
        # the original TabDDPM paper. Set to 5.0 to enable.
        min_snr_gamma: float = 0.0,
        # When True, each training batch is sampled with per-row
        # probabilities inversely proportional to that row's class
        # frequency, so the minority-class conditioning path gets
        # equal gradient exposure. Default False — uniform-over-rows
        # sampling, matching the Yandex TabDDPM reference.
        class_balanced_sampling: bool = False,
        p_uncond: float = 0.1,
        # Default 1.0 — plain conditional sampling, no CFG amplification.
        # CFG at scale > 1 multiplies the (cond − uncond) ε prediction,
        # which on this dataset pushed x0 past the ±4σ clip boundary
        # and caused class-conditional mode collapse. Sample-time only;
        # can be bumped post-load via `iface._model.guidance_scale`.
        guidance_scale: float = 1.0,
        # EDM-style σ schedule (Karras et al. 2022).  When True, fit and
        # generate both switch from the cosine-β DDPM path to log-normal
        # σ sampling + Heun-2 ODE sampling.  Trained models must be
        # generated with the same setting — the saved metadata records
        # this so load_from() rebuilds the right path.
        use_edm_schedule: bool = False,
        edm_p_mean: float = DEFAULT_P_MEAN,
        edm_p_std: float = DEFAULT_P_STD,
        edm_sigma_min: float = DEFAULT_SIGMA_MIN,
        edm_sigma_max: float = DEFAULT_SIGMA_MAX,
        edm_rho: float = DEFAULT_RHO,
        edm_sigma_data: float = DEFAULT_SIGMA_DATA,
        # Cross-feature self-attention head (Step 3 of the GAN improvement
        # plan). Default True — added as a residual contribution to the
        # main MLP, so the model degrades to the pre-Step-3 architecture
        # exactly when set False.
        use_attention: bool = True,
        attn_d_feature: int = 32,
        attn_n_heads: int = 4,
        verbose: bool = True,
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_pairs = num_pairs
        self.pair_names: Optional[List[str]] = None
        self.d_model = d_model
        self.d_layers = tuple(d_layers)
        self.dropout = dropout
        self.use_attention = use_attention
        self.attn_d_feature = attn_d_feature
        self.attn_n_heads = attn_n_heads
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
        self.class_balanced_sampling = class_balanced_sampling
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale
        self.use_edm_schedule = use_edm_schedule
        self.edm_p_mean = edm_p_mean
        self.edm_p_std = edm_p_std
        self.edm_sigma_min = edm_sigma_min
        self.edm_sigma_max = edm_sigma_max
        self.edm_rho = edm_rho
        self.edm_sigma_data = edm_sigma_data
        self.verbose = verbose

        # Feature stats populated by fit(); used by _postprocess.
        # Per-feature z-score (mean / std) rather than minmax because
        # diffusion's forward process pushes data into x_T ≈ N(0, I) —
        # which only makes sense if each feature has roughly unit
        # variance.  Heterogeneous post-minmax variances cause low-σ
        # features to drown in the unit-variance noise and high-σ
        # features to dominate the loss.
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

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
            num_pairs=self.num_pairs,
            use_attention=self.use_attention,
            attn_d_feature=self.attn_d_feature,
            attn_n_heads=self.attn_n_heads,
        )
        self._ema_mlp = _TabDDPMMLP(
            self.num_features, self.num_classes,
            d_model=self.d_model, d_layers=self.d_layers,
            dropout=self.dropout,
            num_pairs=self.num_pairs,
            use_attention=self.use_attention,
            attn_d_feature=self.attn_d_feature,
            attn_n_heads=self.attn_n_heads,
        )

    # ---------- training ---------- #

    # Z-score outlier-clip threshold.  Real data outside this band is
    # rare (< 0.01% under a normal assumption) and almost always an
    # outlier, so we clip during training so the model isn't asked to
    # spend capacity on the tails.  Same threshold applies at sample
    # time to keep generated values inside the region the model has
    # actually seen.
    _ZSCORE_CLIP: float = 4.0

    # Inference-time dispersion scale on the FINAL (de-z-scored) output. The
    # trained denoiser is mode-seeking and UNDER-disperses (σ_syn/σ_real < 1) —
    # teaching the classifier a too-tight Buy/Sell region so it rejects
    # net-winning marginal entries (GAN_TODO #5 diagnosis). The under-dispersion
    # is baked into the model's score (sampler stochasticity can't fix it — it
    # denoises back to the modes), so we widen the OUTPUT: scale each feature
    # around its per-class mean by this factor. Scaling centered features by one
    # factor leaves correlations EXACTLY unchanged (validated) and applies after
    # the ±_ZSCORE_CLIP + inverse z-score, so no clip truncation. 1.0 = off.
    # Pushed on via _apply_gan_inference_overrides(gan_inference_dispersion_scale).
    _DISPERSION_SCALE: float = 1.0

    def _zscore_fit(self, data: np.ndarray) -> np.ndarray:
        """Per-feature z-score normalisation with outlier clipping.

        Each feature gets approximately unit variance, which aligns
        with diffusion's structural assumption (x_T ≈ N(0, I)).  A
        small std floor avoids divide-by-zero on near-constant columns.
        Outliers beyond ±_ZSCORE_CLIP σ are clipped to keep training
        data in a bounded region.
        """
        self.feature_mean = data.mean(axis=0).astype(np.float32)
        std = data.std(axis=0).astype(np.float32)
        self.feature_std = np.maximum(std, 1e-6).astype(np.float32)
        z = (data - self.feature_mean) / self.feature_std
        return np.clip(z, -self._ZSCORE_CLIP, self._ZSCORE_CLIP).astype(np.float32)

    def _zscore_invert(self, z: np.ndarray) -> np.ndarray:
        """Inverse of _zscore_fit — maps standardised samples back to
        the original feature ranges."""
        return z * self.feature_std + self.feature_mean

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        categorical_columns: Optional[Sequence[str]] = None,
        pair_labels: Optional[np.ndarray] = None,
        pair_names: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> None:
        """Train the diffusion model.

        Args:
            data:                (N, F) float32 — continuous features only.
            labels:              (N, C) one-hot float32.
            categorical_columns: Warned about and dropped (the v1 MLX
                                 TabDDPM is continuous-only — same policy
                                 as MLX CTAB-GAN).
            pair_labels:         Optional (N,) int array — pair id per row.
                                 When provided, enables pair conditioning:
                                 model learns per-(class, pair) joint
                                 distributions instead of class-only.
            pair_names:          Optional list of pair display names,
                                 stored in metadata so callers can map a
                                 pair string → pair id at generate time.
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

        # Pair-conditioning setup.  When pair_labels is provided, derive
        # num_pairs and store the name list in metadata.  The model is
        # then built with a pair_embed of size num_pairs.
        if pair_labels is not None:
            pair_labels_np = np.asarray(pair_labels, dtype=np.int32)
            if pair_labels_np.shape != (data.shape[0],):
                raise ValueError(
                    f"pair_labels shape must be ({data.shape[0]},); "
                    f"got {pair_labels_np.shape}"
                )
            derived = int(pair_labels_np.max()) + 1 if pair_labels_np.size else 0
            self.num_pairs = max(
                derived, len(pair_names) if pair_names else 0
            )
            if pair_names is not None:
                self.pair_names = list(pair_names)
            if self.verbose:
                print(
                    f"[TabDDPMMLX] pair conditioning enabled — "
                    f"num_pairs={self.num_pairs}"
                )
        else:
            pair_labels_np = None

        # Lazy-init the model now that we know dimensions.  Rebuild when
        # the live model's num_pairs no longer matches self.num_pairs —
        # that happens when the ctor built it at num_pairs=0 but fit()
        # was then handed pair_labels and bumped self.num_pairs above.
        if self.num_features == 0:
            self.num_features = data.shape[1]
        if self.num_classes == 0:
            self.num_classes = labels.shape[1]
        live_num_pairs = (
            getattr(self._mlp, "num_pairs", 0) if self._mlp is not None else 0
        )
        if self._mlp is None or live_num_pairs != self.num_pairs:
            self._build_models()

        data_norm = self._zscore_fit(data)
        class_idx_np = labels.argmax(axis=1).astype(np.int32)
        N = data_norm.shape[0]

        # Move to MLX arrays once.
        data_mx = mx.array(data_norm)
        class_idx_mx = mx.array(class_idx_np)
        pair_idx_mx = (
            mx.array(pair_labels_np) if pair_labels_np is not None else None
        )

        optimizer = optim.AdamW(learning_rate=self.learning_rate,
                                weight_decay=self.weight_decay)

        # Class-balanced batch-sampling weights.  When enabled, each
        # batch is sampled with per-row probabilities inversely
        # proportional to that row's class frequency, so each class
        # contributes equal total probability; within a class the
        # samples remain uniform.  When disabled, falls back to the
        # uniform-over-rows sampler the original TabDDPM uses.
        if self.class_balanced_sampling:
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
        else:
            per_sample_weights = None

        use_min_snr = self.min_snr_gamma > 0.0

        def loss_fn_ddpm(model: _TabDDPMMLP, x0: mx.array, t: mx.array,
                         noise: mx.array, cls: mx.array,
                         pair: Optional[mx.array] = None) -> mx.array:
            """Cosine-β DDPM loss (the default path)."""
            x_t = q_sample(x0, t, noise, self._sched)
            eps_hat = model(x_t, t, cls, pair_idx=pair)
            sq_err = (eps_hat - noise) ** 2
            if not use_min_snr:
                return mx.mean(sq_err)
            per_sample_sq_err = mx.mean(sq_err, axis=-1)
            ac_t = self._sched.alphas_cumprod[t]
            one_minus_ac = mx.maximum(1.0 - ac_t, 1e-8)
            snr_t = ac_t / one_minus_ac
            weights = mx.minimum(snr_t, self.min_snr_gamma) / snr_t
            return mx.mean(weights * per_sample_sq_err)

        def loss_fn_edm(model: _TabDDPMMLP, x0: mx.array, sigma: mx.array,
                        noise: mx.array, cls: mx.array,
                        pair: Optional[mx.array] = None) -> mx.array:
            """EDM-style σ-schedule loss with ε-prediction + c_in
            preconditioning.

            Forward: x_σ = x_0 + σ · ε.  We then pass the network
            ``c_in · x_σ`` rather than x_σ directly, where
            ``c_in = 1 / sqrt(σ² + σ_data²)`` (EDM Eq. 7) — this keeps
            the input to the MLP at roughly unit magnitude across all σ,
            so the network sees in-distribution inputs whether σ ≈ 0
            or σ ≈ σ_max.  Time conditioning is c_noise = 0.25·ln(σ).
            Loss is plain MSE on ε (no output preconditioning — we keep
            ε-prediction).
            """
            x_sigma = x0 + sigma[:, None] * noise
            sigma_data = self.edm_sigma_data
            c_in = 1.0 / mx.sqrt(sigma**2 + sigma_data**2)
            c_noise = 0.25 * mx.log(sigma)
            eps_hat = model(c_in[:, None] * x_sigma, c_noise, cls, pair_idx=pair)
            return mx.mean((eps_hat - noise) ** 2)

        loss_fn = loss_fn_edm if self.use_edm_schedule else loss_fn_ddpm
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
                pair = pair_idx_mx[idx_mx] if pair_idx_mx is not None else None

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

                noise = mx.random.normal((self.batch_size, self.num_features))
                if self.use_edm_schedule:
                    # Log-normal σ sampling per EDM Eq. 5.
                    t_or_sigma = sample_log_normal_sigma(
                        self.batch_size,
                        p_mean=self.edm_p_mean,
                        p_std=self.edm_p_std,
                    )
                else:
                    # Uniform timestep sampling per standard DDPM.
                    t_or_sigma = mx.random.randint(
                        0, self.num_timesteps, (self.batch_size,)
                    )

                loss, grads = loss_and_grad(self._mlp, x0, t_or_sigma, noise, cls, pair)
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

    def generate(
        self,
        n: int,
        one_hot: np.ndarray,
        pair_label: Optional[int] = None,
    ) -> np.ndarray:
        """Sample n synthetic rows conditioned on `one_hot` (and optionally
        on a pair id).

        Args:
            n:          Number of samples.
            one_hot:    (n, num_classes) float32.
            pair_label: Optional int — pair id to condition on.  Required
                        when the model was trained with pair conditioning
                        (num_pairs > 0); ignored otherwise.  If the model
                        has pair conditioning but no label is supplied,
                        pairs are sampled uniformly from the training set.

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

        # Pair conditioning: if the model has a pair_embed, every forward
        # pass needs a pair_idx.  Use the supplied pair_label when given,
        # otherwise sample uniformly from training pair ids (a soft
        # "average over pairs" — useful when the caller doesn't care which
        # pair the synth samples come from).
        if self.num_pairs > 0:
            if pair_label is not None:
                pair_idx = mx.full((n,), int(pair_label), dtype=mx.int32)
            else:
                pair_ids = np.random.randint(0, self.num_pairs, n).astype(np.int32)
                pair_idx = mx.array(pair_ids)
        else:
            pair_idx = None

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

        # The model's time-conditioning input differs by schedule:
        #   DDPM path: integer timestep t (passed straight through).
        #   EDM  path: c_noise(σ) = 0.25 · ln(σ) per Karras et al. Eq. 7,
        #             and input scaled by c_in = 1/sqrt(σ² + σ_data²)
        #             to keep MLP inputs at unit magnitude across all σ
        #             — without this, σ-magnitude inputs to the network
        #             go non-finite for σ values it never saw during
        #             training.
        if self.use_edm_schedule:
            sigma_data_val = self.edm_sigma_data

            def _time_cond(t_or_sigma: mx.array) -> mx.array:
                return 0.25 * mx.log(t_or_sigma)

            def _scale_input(x: mx.array, sigma: mx.array) -> mx.array:
                c_in = 1.0 / mx.sqrt(sigma**2 + sigma_data_val**2)
                return c_in[:, None] * x
        else:
            def _time_cond(t_or_sigma: mx.array) -> mx.array:
                return t_or_sigma

            def _scale_input(x: mx.array, sigma: mx.array) -> mx.array:
                return x

        if guidance == 1.0:
            def model_fn(x_t: mx.array, t: mx.array, cond: mx.array) -> mx.array:
                return ema(_scale_input(x_t, t), _time_cond(t), cond, pair_idx=pair_idx)
        else:
            null_idx = mx.full((n,), self.num_classes, dtype=mx.int32)

            def model_fn(x_t: mx.array, t: mx.array, cond: mx.array) -> mx.array:
                x_in = _scale_input(x_t, t)
                c = _time_cond(t)
                eps_cond = ema(x_in, c, cond, pair_idx=pair_idx)
                eps_uncond = ema(x_in, c, null_idx, pair_idx=pair_idx)
                return eps_uncond + guidance * (eps_cond - eps_uncond)

        try:
            if self.use_edm_schedule:
                sigmas = build_sigma_schedule(
                    self.num_sample_steps,
                    sigma_min=self.edm_sigma_min,
                    sigma_max=self.edm_sigma_max,
                    rho=self.edm_rho,
                )
                x0_mx = heun_sample(
                    model_fn=model_fn,
                    shape=(n, self.num_features),
                    cond=class_idx,
                    sigmas=sigmas,
                )
            else:
                x0_mx = ddim_sample(
                    model_fn=model_fn,
                    shape=(n, self.num_features),
                    cond=class_idx,
                    sched=self._sched,
                    num_steps=self.num_sample_steps,
                )
        finally:
            ema.train()

        # _postprocess: clip to the training-time z-score band (±4σ),
        # then inverse z-score back to original feature ranges.
        x0_np = np.clip(np.asarray(x0_mx), -self._ZSCORE_CLIP, self._ZSCORE_CLIP)
        x0_np = self._zscore_invert(x0_np)
        # Optional dispersion widening on the FINAL output (counteract the
        # denoiser's under-dispersion). generate() is called per-class, so the
        # batch mean is the class mean; scaling around it by one factor widens σ
        # while leaving feature correlations exactly unchanged.
        scale = float(getattr(self, "_DISPERSION_SCALE", 1.0))
        if scale != 1.0 and x0_np.shape[0] > 1:
            mu = x0_np.mean(axis=0, keepdims=True)
            x0_np = mu + scale * (x0_np - mu)
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
            "num_pairs":        self.num_pairs,
            "pair_names":       self.pair_names,
            "d_model":          self.d_model,
            "d_layers":         list(self.d_layers),
            "dropout":          self.dropout,
            "use_attention":    self.use_attention,
            "attn_d_feature":   self.attn_d_feature,
            "attn_n_heads":     self.attn_n_heads,
            "num_timesteps":    self.num_timesteps,
            "num_sample_steps": self.num_sample_steps,
            "p_uncond":         self.p_uncond,
            "guidance_scale":   self.guidance_scale,
            "use_edm_schedule": self.use_edm_schedule,
            "edm_p_mean":       self.edm_p_mean,
            "edm_p_std":        self.edm_p_std,
            "edm_sigma_min":    self.edm_sigma_min,
            "edm_sigma_max":    self.edm_sigma_max,
            "edm_rho":          self.edm_rho,
            "edm_sigma_data":   self.edm_sigma_data,
            "feature_mean":     np.asarray(self.feature_mean, dtype=np.float32),
            "feature_std":      np.asarray(self.feature_std, dtype=np.float32),
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
            num_pairs=int(metadata.get("num_pairs", 0)),
            d_model=int(metadata.get("d_model", 256)),
            d_layers=tuple(metadata.get("d_layers", (256, 256))),
            dropout=float(metadata.get("dropout", 0.0)),
            num_timesteps=int(metadata.get("num_timesteps", 1000)),
            num_sample_steps=int(metadata.get("num_sample_steps", 50)),
            p_uncond=float(metadata.get("p_uncond", 0.1)),
            guidance_scale=float(metadata.get("guidance_scale", 1.0)),
            use_edm_schedule=bool(metadata.get("use_edm_schedule", False)),
            edm_p_mean=float(metadata.get("edm_p_mean", DEFAULT_P_MEAN)),
            edm_p_std=float(metadata.get("edm_p_std", DEFAULT_P_STD)),
            edm_sigma_min=float(metadata.get("edm_sigma_min", DEFAULT_SIGMA_MIN)),
            edm_sigma_max=float(metadata.get("edm_sigma_max", DEFAULT_SIGMA_MAX)),
            edm_rho=float(metadata.get("edm_rho", DEFAULT_RHO)),
            edm_sigma_data=float(metadata.get("edm_sigma_data", DEFAULT_SIGMA_DATA)),
            # Step 3 fields default to False/legacy defaults for backwards
            # compatibility with models saved before the attention head
            # was introduced.
            use_attention=bool(metadata.get("use_attention", False)),
            attn_d_feature=int(metadata.get("attn_d_feature", 32)),
            attn_n_heads=int(metadata.get("attn_n_heads", 4)),
            verbose=False,
        )
        instance._ema_mlp.load_weights(weights_p)
        instance.feature_mean = np.asarray(metadata["feature_mean"], dtype=np.float32)
        instance.feature_std = np.asarray(metadata["feature_std"], dtype=np.float32)
        pair_names = metadata.get("pair_names")
        instance.pair_names = list(pair_names) if pair_names is not None else None
        return instance, metadata
