"""
WGAN-GP (single-task) backends — TF and MLX.

The trainers (``balance_with_wgan_gp``, ``balance_with_wgan_mlx``) are
function-style entry points that return ``(generated, augmented_labels,
gan_model)`` rather than a class with ``.fit()``.  Each wrapper here
calls the underlying function with ``_return_model=True``,
``augmentation_target_ratio=0`` (we only train, the caller asks for
samples later via ``generate()``), and stashes the returned model.

Save / load:

    * MLX: ``WGANMLX.save(path)`` writes the generator's weights
      (``wgan_gen_mlx.safetensors``); the backend separately writes
      ``wgangp_metadata.pkl`` with the constructor params so that
      ``load()`` can reconstruct the model class + load weights.

    * TF: ``_save_wgan_model`` / ``_load_wgan_model`` (module-level
      helpers in ``df_wgan_gp``) handle the full lifecycle including
      a single ``wgangp_metadata.pkl`` + Keras weights file.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from GANs.GANBackend import GANBackend, register_backend
from GANs.GANType import GANType


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mlx_available() -> bool:
    """Lazy import of ``GANs.GANInterface._HAS_MLX`` so test patches still
    take effect.  Same pattern used by the CTAB MLX backend."""
    from GANs.GANInterface import _HAS_MLX
    return _HAS_MLX


def _data_to_2d(data: Any) -> np.ndarray:
    """``balance_with_wgan_*`` expects 2-D tabular input.  If the caller
    passed a 3-D tensor with seq_len=1, squeeze.  Other seq_lens raise
    because tabular WGAN can't represent sequence structure."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[1] != 1:
            raise ValueError(
                f"Single-task WGAN expects (N, F) tabular data; got shape "
                f"{arr.shape}.  Use MT_WGAN for sequential input."
            )
        arr = arr[:, 0, :]
    return arr


def _build_balance_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build the kwargs dict passed to ``balance_with_wgan_*``.

    Matches the contract the previous ``GANInterface._fit_wgan`` set up:
    save_path=None (caller saves explicitly), assess_quality=False,
    augmentation_target_ratio=0 (train only, no in-fit generation),
    seq_len defaulted to 1.
    """
    config = dict(kwargs)
    config.setdefault("seq_len", 1)
    config["save_path"] = None
    config["assess_quality"] = False
    config["augmentation_target_ratio"] = 0.0
    return config


# ---------------------------------------------------------------------------
# WGAN-GP (TF)
# ---------------------------------------------------------------------------


@register_backend
class WGANTFBackend(GANBackend):
    """TF backend for single-task WGAN-GP."""

    GAN_TYPE = GANType.WGAN
    PREFERS_MLX = False

    def __init__(self, model: Optional[Any] = None) -> None:
        self._model = model

    @classmethod
    def is_available(cls) -> bool:
        # No eager TF import — see the docstring on CTABGANTFBackend.
        return True

    # ------------------------------------------------------------------
    # fit / generate
    # ------------------------------------------------------------------

    def fit(
        self,
        data: Any,
        labels: Any,
        categorical_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        from GANs.df_wgan_gp import balance_with_wgan_gp  # noqa: E402

        data_2d = _data_to_2d(data)
        labels_f32 = np.asarray(labels, dtype=np.float32)
        config = _build_balance_config(kwargs)

        _, _, gan = balance_with_wgan_gp(
            data_2d, labels_f32, _return_model=True, **config
        )
        self._model = gan

    def generate(self, n: int, **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError(
                "WGANTFBackend.generate called before fit/load — no model"
            )
        one_hot = kwargs.get("one_hot")
        if one_hot is None:
            raise ValueError(
                "generate() for WGAN requires keyword argument one_hot=<np.ndarray>"
            )
        return self._model.generate(n, one_hot)

    # ------------------------------------------------------------------
    # save / load — module-level helpers in df_wgan_gp
    # ------------------------------------------------------------------

    def save(self, save_path: str, **extra_metadata: Any) -> None:
        if self._model is None:
            raise RuntimeError("WGANTFBackend.save called before fit — no model")

        from GANs.df_wgan_gp import _save_wgan_model  # noqa: E402

        meta = dict(getattr(self._model, "_interface_metadata", {}))
        meta.update(extra_metadata)
        _save_wgan_model(self._model, meta, save_path)

    @classmethod
    def load(cls, save_path: str) -> Tuple["WGANTFBackend", Dict[str, Any]]:
        from GANs.df_wgan_gp import _load_wgan_model  # noqa: E402

        gan, metadata = _load_wgan_model(save_path)
        if gan is None:
            raise FileNotFoundError(
                f"No saved WGAN model found at {save_path}"
            )
        instance = cls()
        instance._model = gan
        instance._model._interface_metadata = metadata or {}
        return instance, metadata or {}


# ---------------------------------------------------------------------------
# WGAN-GP (MLX)
# ---------------------------------------------------------------------------

# Identifying file for "this directory holds an MLX-format WGAN".  Both
# the safetensors weights and the metadata pickle must be present for
# load() to succeed; we probe the metadata pickle since it's the smaller
# read.
_MLX_META_FILENAME = "wgangp_metadata.pkl"
_MLX_WEIGHTS_FILENAME = "wgan_gen_mlx.safetensors"


@register_backend
class WGANMLXBackend(GANBackend):
    """MLX backend for single-task WGAN-GP."""

    GAN_TYPE = GANType.WGAN
    PREFERS_MLX = True

    def __init__(self, model: Optional[Any] = None) -> None:
        self._model = model

    @classmethod
    def is_available(cls) -> bool:
        return _mlx_available()

    # ------------------------------------------------------------------
    # fit / generate
    # ------------------------------------------------------------------

    def fit(
        self,
        data: Any,
        labels: Any,
        categorical_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        from GANs.df_wgan_mlx import balance_with_wgan_mlx  # noqa: E402

        data_2d = _data_to_2d(data)
        labels_f32 = np.asarray(labels, dtype=np.float32)
        config = _build_balance_config(kwargs)

        _, _, gan = balance_with_wgan_mlx(
            data_2d, labels_f32, _return_model=True, **config
        )
        self._model = gan

    def generate(self, n: int, **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError(
                "WGANMLXBackend.generate called before fit/load — no model"
            )
        one_hot = kwargs.get("one_hot")
        if one_hot is None:
            raise ValueError(
                "generate() for WGAN requires keyword argument one_hot=<np.ndarray>"
            )
        return self._model.generate(n, one_hot)

    # ------------------------------------------------------------------
    # save / load — split between WGANMLX (weights) and a sidecar pickle
    # ------------------------------------------------------------------

    def save(self, save_path: str, **extra_metadata: Any) -> None:
        if self._model is None:
            raise RuntimeError("WGANMLXBackend.save called before fit — no model")

        os.makedirs(save_path, exist_ok=True)
        # WGANMLX writes its own safetensors file.
        self._model.save(save_path)

        # The interface owns the metadata pickle so load() can
        # reconstruct the WGANMLX class with the right ctor args.
        meta = {
            "num_features": self._model.num_features,
            "num_classes":  self._model.num_classes,
            "latent_dim":   self._model.latent_dim,
            "seq_len":      1,
        }
        meta.update(extra_metadata)
        with open(os.path.join(save_path, _MLX_META_FILENAME), "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, save_path: str) -> Tuple["WGANMLXBackend", Dict[str, Any]]:
        meta_path = os.path.join(save_path, _MLX_META_FILENAME)
        weights_path = os.path.join(save_path, _MLX_WEIGHTS_FILENAME)

        # Both files must exist for this to be an MLX-format save.
        # Missing either is "not my format" — raise FileNotFoundError so
        # load_with_fallback moves on to the TF backend.
        if not (os.path.exists(meta_path) and os.path.exists(weights_path)):
            raise FileNotFoundError(
                f"No MLX-format WGAN model at {save_path} "
                f"(needs {_MLX_META_FILENAME} + {_MLX_WEIGHTS_FILENAME})"
            )

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        from GANs.df_wgan_mlx import WGANMLX  # noqa: E402

        gan = WGANMLX(
            metadata["num_features"],
            metadata["num_classes"],
            latent_dim=metadata.get("latent_dim", 64),
        )
        gan.load(save_path)

        instance = cls()
        instance._model = gan
        return instance, metadata or {}
