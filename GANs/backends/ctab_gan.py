"""
CTAB-GAN family backends.

Three concrete classes:
  * CTABGANTFBackend       (TF, single-task)
  * CTABGANMLXBackend      (MLX, single-task)
  * MTCTABGANTFBackend     (TF, multi-task — no MLX variant exists)

These wrap the existing ``CTABGANPlus`` / ``CTABGANMLX`` / ``CTABGANPlusMT``
trainer classes — the trainers already follow the
fit / generate / save / load shape, so each backend is mostly an adapter
that partitions caller-supplied kwargs into "constructor args" and
"fit args" the way the trainer expects.

Constructor-key / skip-key sets are lifted verbatim from the previous
``GANInterface._CTAB_CTOR_KEYS`` etc. so behaviour is preserved
bit-for-bit during the migration.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from GANs.GANBackend import GANBackend, register_backend
from GANs.GANType import GANType


# ---------------------------------------------------------------------------
# Constructor / fit kwarg classification
# ---------------------------------------------------------------------------

# Keys that go to the CTABGANPlus / CTABGANPlusMT constructor.  Anything
# in caller_kwargs that isn't here AND isn't in _SKIP_KEYS is forwarded to
# fit() (in practice usually just ``validation_split``).
_CTAB_CTOR_KEYS: frozenset = frozenset({
    "latent_dim", "generator_layers", "discriminator_layers",
    "batch_size", "epochs", "learning_rate", "beta_1", "beta_2",
    "gp_weight", "verbose", "early_stopping_patience",
    "early_stopping_min_delta", "reduce_lr_patience", "reduce_lr_factor",
    "reduce_lr_min", "pac", "monitor_metric", "eval_frequency",
    "eval_num_samples", "random_seed", "integer_columns",
    # MT-only extras
    "use_cnn", "use_auxiliary",
})

# Keys that belong to other backends or to the surrounding strategy
# config (CreateGANBase, etc.) and that the CTAB constructor doesn't
# accept.  Silently dropped.
_CTAB_SKIP_KEYS: frozenset = frozenset({
    "save_path", "augmentation_target_ratio", "task_target_ratios",
    "assess_quality", "n_critic", "noise_std", "architecture", "seq_len",
})

# CTAB-GAN MLX (single-task and multi-task) uses a slimmer constructor —
# drop anything not in this set.  Both MLX classes accept the same params
# (the multi-task class adds eval-quality early stopping / LR-reduce
# alongside the single-task version).
_CTAB_MLX_CTOR_KEYS: frozenset = frozenset({
    "latent_dim", "hidden_dim", "epochs", "batch_size", "n_critic",
    "learning_rate", "gp_weight",
    "early_stopping_patience", "early_stopping_min_delta",
    "reduce_lr_patience", "reduce_lr_factor", "reduce_lr_min",
    "eval_frequency", "eval_num_samples", "monitor_metric",
    "random_seed", "verbose",
})


def _split_ctab_kwargs(
    kwargs: Dict[str, Any],
    ctor_keys: frozenset = _CTAB_CTOR_KEYS,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Partition caller kwargs into (ctor_kwargs, fit_kwargs)."""
    ctor = {k: v for k, v in kwargs.items() if k in ctor_keys}
    fit = {
        k: v
        for k, v in kwargs.items()
        if k not in ctor_keys and k not in _CTAB_SKIP_KEYS
    }
    return ctor, fit


# ---------------------------------------------------------------------------
# Helpers shared by every backend in this module
# ---------------------------------------------------------------------------


def _mlx_available() -> bool:
    """Whether the MLX backend should be considered available.

    Defers to ``GANs.GANInterface._HAS_MLX`` (lazy import to break the
    backends ↔ GANInterface cycle) so that tests can patch a single flag
    and have every consumer see the change consistently.  Direct
    ``import mlx.core`` here would bypass that patch.
    """
    from GANs.GANInterface import _HAS_MLX
    return _HAS_MLX


# ---------------------------------------------------------------------------
# CTAB-GAN+ (TF) — single-task
# ---------------------------------------------------------------------------


@register_backend
class CTABGANTFBackend(GANBackend):
    """TF backend for single-task CTAB-GAN+ (wraps CTABGANPlus)."""

    GAN_TYPE = GANType.CTAB_GAN
    PREFERS_MLX = False

    def __init__(self, model: Optional[Any] = None) -> None:
        self._model = model

    @classmethod
    def is_available(cls) -> bool:
        # TF backends rely on the eventual ``import tensorflow`` inside
        # fit/load to surface a missing-dependency ImportError, matching
        # the pre-refactor behaviour where there was no eager check.
        return True

    # ---------- lifecycle ----------

    def fit(
        self,
        data: Any,
        labels: Any,
        categorical_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        from GANs.df_ctab_gan import CTABGANPlus  # noqa: E402

        ctor_kwargs, fit_kwargs = _split_ctab_kwargs(kwargs)
        self._model = CTABGANPlus(**ctor_kwargs)
        self._model.fit(
            dataframe=data,
            labels=labels,
            categorical_columns=categorical_columns or [],
            **fit_kwargs,
        )

    def generate(self, n: int, **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError(
                "CTABGANTFBackend.generate called before fit/load — no model"
            )
        # Historically GANInterface allowed callers to pass num_samples=
        # explicitly; honour that for backward compat.
        num_samples = kwargs.pop("num_samples", n)
        return self._model.generate(num_samples=num_samples, **kwargs)

    def save(self, save_path: str, **extra_metadata: Any) -> None:
        if self._model is None:
            raise RuntimeError("CTABGANTFBackend.save called before fit — no model")
        self._model.save(save_path, **extra_metadata)

    @classmethod
    def load(cls, save_path: str) -> Tuple["CTABGANTFBackend", Dict[str, Any]]:
        from GANs.df_ctab_gan import CTABGANPlus  # noqa: E402

        instance = cls()
        instance._model = CTABGANPlus()
        metadata = instance._model.load(save_path)
        return instance, metadata or {}


# ---------------------------------------------------------------------------
# CTAB-GAN+ (MLX) — single-task
# ---------------------------------------------------------------------------


@register_backend
class CTABGANMLXBackend(GANBackend):
    """MLX backend for single-task CTAB-GAN+ (wraps CTABGANMLX)."""

    GAN_TYPE = GANType.CTAB_GAN
    PREFERS_MLX = True

    def __init__(self, model: Optional[Any] = None) -> None:
        self._model = model

    @classmethod
    def is_available(cls) -> bool:
        return _mlx_available()

    # ---------- lifecycle ----------

    def fit(
        self,
        data: Any,
        labels: Any,
        categorical_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        from GANs.df_ctab_gan_mlx import CTABGANMLX  # noqa: E402

        # MLX has a slimmer constructor — only forward keys it actually accepts.
        mlx_kwargs = {k: v for k, v in kwargs.items() if k in _CTAB_MLX_CTOR_KEYS}
        self._model = CTABGANMLX(**mlx_kwargs)
        self._model.fit(
            dataframe=data,
            labels=labels,
            categorical_columns=categorical_columns or [],
        )

    def generate(self, n: int, **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError(
                "CTABGANMLXBackend.generate called before fit/load — no model"
            )
        num_samples = kwargs.pop("num_samples", n)
        return self._model.generate(num_samples=num_samples, **kwargs)

    def save(self, save_path: str, **extra_metadata: Any) -> None:
        if self._model is None:
            raise RuntimeError("CTABGANMLXBackend.save called before fit — no model")
        self._model.save(save_path, **extra_metadata)

    @classmethod
    def load(cls, save_path: str) -> Tuple["CTABGANMLXBackend", Dict[str, Any]]:
        # MLX format is identified by the presence of metadata_mlx.pkl.
        # If it's missing, this backend can't honour the load — caller
        # should fall through to the TF backend (resolve_backend handles
        # that ordering automatically).
        if not os.path.exists(os.path.join(save_path, "metadata_mlx.pkl")):
            raise FileNotFoundError(
                f"No MLX-format CTAB-GAN model at {save_path} "
                f"(metadata_mlx.pkl missing)"
            )

        from GANs.df_ctab_gan_mlx import CTABGANMLX  # noqa: E402

        instance = cls()
        instance._model = CTABGANMLX()
        metadata = instance._model.load(save_path)
        return instance, metadata or {}


# ---------------------------------------------------------------------------
# Multi-task CTAB-GAN+ (TF only — no MLX variant)
# ---------------------------------------------------------------------------


@register_backend
class MTCTABGANTFBackend(GANBackend):
    """TF backend for multi-task CTAB-GAN+ (wraps CTABGANPlusMT)."""

    GAN_TYPE = GANType.MT_CTAB_GAN
    PREFERS_MLX = False

    def __init__(self, model: Optional[Any] = None) -> None:
        self._model = model

    @classmethod
    def is_available(cls) -> bool:
        # TF backends rely on the eventual ``import tensorflow`` inside
        # fit/load to surface a missing-dependency ImportError, matching
        # the pre-refactor behaviour where there was no eager check.
        return True

    # ---------- lifecycle ----------

    def fit(
        self,
        data: Any,
        labels: Any,
        categorical_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        from GANs.df_mt_ctab_gan import CTABGANPlusMT  # noqa: E402

        ctor_kwargs, fit_kwargs = _split_ctab_kwargs(kwargs)
        self._model = CTABGANPlusMT(**ctor_kwargs)
        self._model.fit(
            dataframe=data,
            labels=labels,
            categorical_columns=categorical_columns or [],
            **fit_kwargs,
        )

    def generate(self, n: int, **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError(
                "MTCTABGANTFBackend.generate called before fit/load — no model"
            )
        num_samples = kwargs.pop("num_samples", n)
        return self._model.generate(num_samples=num_samples, **kwargs)

    def save(self, save_path: str, **extra_metadata: Any) -> None:
        if self._model is None:
            raise RuntimeError("MTCTABGANTFBackend.save called before fit — no model")
        self._model.save(save_path, **extra_metadata)

    @classmethod
    def load(cls, save_path: str) -> Tuple["MTCTABGANTFBackend", Dict[str, Any]]:
        from GANs.df_mt_ctab_gan import CTABGANPlusMT  # noqa: E402

        instance = cls()
        instance._model = CTABGANPlusMT()
        metadata = instance._model.load(save_path)
        return instance, metadata or {}


# ---------------------------------------------------------------------------
# Multi-task CTAB-GAN+ (MLX)
# ---------------------------------------------------------------------------


@register_backend
class MTCTABGANMLXBackend(GANBackend):
    """MLX backend for multi-task CTAB-GAN+ (wraps CTABGANMLXMT).

    Continuous-only — categorical_columns are silently dropped (the
    underlying class warns).  Use the TF backend if you need
    categorical support for the multi-task variant.
    """

    GAN_TYPE = GANType.MT_CTAB_GAN
    PREFERS_MLX = True

    def __init__(self, model: Optional[Any] = None) -> None:
        self._model = model

    @classmethod
    def is_available(cls) -> bool:
        return _mlx_available()

    # ---------- lifecycle ----------

    def fit(
        self,
        data: Any,
        labels: Any,
        categorical_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        from GANs.df_mt_ctab_gan_mlx import CTABGANMLXMT  # noqa: E402

        mlx_kwargs = {k: v for k, v in kwargs.items() if k in _CTAB_MLX_CTOR_KEYS}
        self._model = CTABGANMLXMT(**mlx_kwargs)
        self._model.fit(
            dataframe=data,
            labels=labels,
            categorical_columns=categorical_columns or [],
        )

    def generate(self, n: int, **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError(
                "MTCTABGANMLXBackend.generate called before fit/load — no model"
            )
        num_samples = kwargs.pop("num_samples", n)
        return self._model.generate(num_samples=num_samples, **kwargs)

    def save(self, save_path: str, **extra_metadata: Any) -> None:
        if self._model is None:
            raise RuntimeError("MTCTABGANMLXBackend.save called before fit — no model")
        self._model.save(save_path, **extra_metadata)

    @classmethod
    def load(cls, save_path: str) -> Tuple["MTCTABGANMLXBackend", Dict[str, Any]]:
        # MLX format is identified by the presence of metadata_mlx.pkl —
        # same convention as CTABGANMLXBackend.  Missing file → fall
        # through to the TF backend via resolve_backend's fallback chain.
        if not os.path.exists(os.path.join(save_path, "metadata_mlx.pkl")):
            raise FileNotFoundError(
                f"No MLX-format MT CTAB-GAN model at {save_path} "
                f"(metadata_mlx.pkl missing)"
            )

        from GANs.df_mt_ctab_gan_mlx import CTABGANMLXMT  # noqa: E402

        instance = cls()
        instance._model = CTABGANMLXMT()
        metadata = instance._model.load(save_path)
        return instance, metadata or {}
