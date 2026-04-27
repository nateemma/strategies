"""
Multi-task WGAN-GP backends — TF and MLX.

Same function-style training entry points as single-task WGAN
(``balance_with_mt_wgan_gp``, ``balance_with_mt_wgan_mlx``), with two
multi-task-specific wrinkles the wrappers handle:

  * Labels are a ``Dict[str, np.ndarray]`` (one per task: trading,
    regime, risk, momentum, flow, profit) rather than a flat one-hot
    array.

  * The MLX save path tags filenames with a ``_s{seq_len}`` suffix
    when ``seq_len > 1`` (e.g. ``mt_wgan_meta_mlx_s16.pkl``).  Load
    must glob for whichever suffix is present, since the caller
    invoking ``load_with_fallback`` doesn't know seq_len up front.
    This is the bug we hit pre-refactor — see commit message of the
    'GANInterface MT_WGAN MLX load glob fix'.

  * The TF model has Keras ``TrackedDict`` attributes that aren't
    picklable; we sanitise them to plain dicts/floats before
    handing the metadata to ``_save_mt_wgan_model``.
"""

from __future__ import annotations

import glob
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
    from GANs.GANInterface import _HAS_MLX
    return _HAS_MLX


def _build_mt_balance_config(kwargs: Dict[str, Any], labels: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Build kwargs for ``balance_with_mt_wgan_*``.

    Mirrors the contract the previous ``GANInterface._fit_mt_wgan`` set
    up: ``save_path=None`` (caller saves explicitly), ``assess_quality=False``,
    and ``task_target_ratios`` zeroed for every task so the in-fit
    augmentation loop is skipped (we train only; ``generate()`` synthesises
    samples on demand).
    """
    config = dict(kwargs)
    config["save_path"] = None
    config["assess_quality"] = False
    config["task_target_ratios"] = {k: 0.0 for k in labels}
    return config


def _sanitise_keras_tracked_dicts(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Keras wraps dict attributes on Model subclasses as TrackedDict,
    which the pickle protocol can't handle.  Convert any TrackedDict-style
    entries we know about into plain Python types before persisting.

    Idempotent: applying twice has no effect.  Applies to the metadata
    that the TF MT-WGAN model attaches as ``_interface_metadata``.
    """
    out = dict(meta)
    if "task_label_dims" in out:
        out["task_label_dims"] = dict(out["task_label_dims"])
    if "task_loss_weights" in out:
        out["task_loss_weights"] = {
            k: float(v) for k, v in out["task_loss_weights"].items()
        }
    return out


# ---------------------------------------------------------------------------
# MT_WGAN-GP (TF)
# ---------------------------------------------------------------------------


@register_backend
class MTWGANTFBackend(GANBackend):
    """TF backend for multi-task WGAN-GP."""

    GAN_TYPE = GANType.MT_WGAN
    PREFERS_MLX = False

    def __init__(self, model: Optional[Any] = None) -> None:
        self._model = model

    @classmethod
    def is_available(cls) -> bool:
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
        from GANs.df_mt_wgan_gp import balance_with_mt_wgan_gp  # noqa: E402

        if not isinstance(labels, dict):
            raise ValueError(
                "MT_WGAN expects labels as Dict[str, np.ndarray] "
                f"(one entry per task); got {type(labels).__name__}"
            )

        data_f32 = np.asarray(data, dtype=np.float32)
        config = _build_mt_balance_config(kwargs, labels)

        _, _, gan = balance_with_mt_wgan_gp(
            data_f32, labels, _return_model=True, **config
        )
        self._model = gan

    def generate(self, n: int, **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError(
                "MTWGANTFBackend.generate called before fit/load — no model"
            )
        task_labels = kwargs.get("task_labels")
        if task_labels is None:
            raise ValueError(
                "generate() for MT_WGAN requires keyword argument task_labels=<dict>"
            )
        return self._model.generate(n, task_labels)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, save_path: str, **extra_metadata: Any) -> None:
        if self._model is None:
            raise RuntimeError("MTWGANTFBackend.save called before fit — no model")

        from GANs.df_mt_wgan_gp import _save_mt_wgan_model  # noqa: E402

        meta = dict(getattr(self._model, "_interface_metadata", {}))
        meta = _sanitise_keras_tracked_dicts(meta)
        meta.update(extra_metadata)
        _save_mt_wgan_model(self._model, meta, save_path)

    @classmethod
    def load(cls, save_path: str) -> Tuple["MTWGANTFBackend", Dict[str, Any]]:
        from GANs.df_mt_wgan_gp import _load_mt_wgan_model  # noqa: E402

        gan, metadata = _load_mt_wgan_model(save_path)
        if gan is None:
            raise FileNotFoundError(
                f"No saved MT_WGAN model found at {save_path}"
            )
        instance = cls()
        instance._model = gan
        instance._model._interface_metadata = metadata or {}
        return instance, metadata or {}


# ---------------------------------------------------------------------------
# MT_WGAN-GP (MLX)
# ---------------------------------------------------------------------------

# Naming convention is documented in MTWGANMLX._get_paths(): files are
# ``mt_wgan_{gen,crit,meta}_mlx[_s{seq_len}].{safetensors,pkl}``.  load()
# globs for the metadata file because we don't know seq_len up front.
_MLX_META_PATTERN = "mt_wgan_meta_mlx*.pkl"


@register_backend
class MTWGANMLXBackend(GANBackend):
    """MLX backend for multi-task WGAN-GP."""

    GAN_TYPE = GANType.MT_WGAN
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
        from GANs.df_mt_wgan_mlx import balance_with_mt_wgan_mlx  # noqa: E402

        if not isinstance(labels, dict):
            raise ValueError(
                "MT_WGAN expects labels as Dict[str, np.ndarray] "
                f"(one entry per task); got {type(labels).__name__}"
            )

        data_f32 = np.asarray(data, dtype=np.float32)
        config = _build_mt_balance_config(kwargs, labels)

        _, _, gan = balance_with_mt_wgan_mlx(
            data_f32, labels, _return_model=True, **config
        )
        self._model = gan

    def generate(self, n: int, **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError(
                "MTWGANMLXBackend.generate called before fit/load — no model"
            )
        task_labels = kwargs.get("task_labels")
        if task_labels is None:
            raise ValueError(
                "generate() for MT_WGAN requires keyword argument task_labels=<dict>"
            )
        return self._model.generate(n, task_labels)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, save_path: str, **extra_metadata: Any) -> None:
        if self._model is None:
            raise RuntimeError("MTWGANMLXBackend.save called before fit — no model")

        # MTWGANMLX.save writes ALL files (gen, crit, meta) using its
        # own _s{seq_len}-suffixed paths.  We supply the metadata dict.
        meta = {
            "seq_len":         self._model.seq_len,
            "num_features":    self._model.num_features,
            "task_label_dims": dict(self._model.task_label_dims),
            "latent_dim":      self._model.latent_dim,
        }
        meta.update(extra_metadata)
        self._model.save(save_path, meta)

    @classmethod
    def load(cls, save_path: str) -> Tuple["MTWGANMLXBackend", Dict[str, Any]]:
        # The MLX save uses a seq_len-suffixed metadata filename
        # (``mt_wgan_meta_mlx_s16.pkl`` for seq_len=16, plain
        # ``mt_wgan_meta_mlx.pkl`` for seq_len=1).  We don't know the
        # suffix up front, so glob.
        candidates = sorted(glob.glob(os.path.join(save_path, _MLX_META_PATTERN)))
        if not candidates:
            raise FileNotFoundError(
                f"No MLX-format MT_WGAN model at {save_path} "
                f"(no files matching {_MLX_META_PATTERN})"
            )

        meta_path = candidates[0]
        if len(candidates) > 1:
            print(
                f"    Multiple MT_WGAN MLX metadata files found in "
                f"{save_path}: {candidates}.  Using {meta_path}."
            )

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        from GANs.df_mt_wgan_mlx import MTWGANMLX  # noqa: E402

        gan = MTWGANMLX(
            seq_len=metadata["seq_len"],
            num_features=metadata["num_features"],
            task_label_dims=metadata["task_label_dims"],
            latent_dim=metadata.get("latent_dim", 64),
        )
        # MTWGANMLX.load reconstructs the per-seq_len paths internally
        # via _get_paths, so it finds the suffixed weight files itself.
        ok, _ = gan.load(save_path)
        if not ok:
            raise RuntimeError(
                f"MTWGANMLX.load() failed for {save_path} "
                f"(metadata file: {meta_path})"
            )

        instance = cls()
        instance._model = gan
        return instance, metadata or {}
