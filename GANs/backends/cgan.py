"""
Conditional GAN (CGAN) backend.

TF-only — no MLX variant exists.  Wraps the function-style training
flow that used to live in ``GANInterface._fit_cgan`` together with the
module-level ``_save_cgan_model`` / ``_load_cgan_model`` helpers from
``GANs.df_cgan``.  Putting all three behind one class lets the
GANInterface facade use the same fit/save/load contract for CGAN as it
does for every other type.

Persisted file layout (from ``_save_cgan_model``):
    cgan_metadata.pkl  +  Keras weights files written by the model
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from GANs.GANBackend import GANBackend, register_backend
from GANs.GANType import GANType


@register_backend
class CGANTFBackend(GANBackend):
    """TF backend for Conditional GAN (DFCGAN)."""

    GAN_TYPE = GANType.CGAN
    PREFERS_MLX = False

    def __init__(self, model: Optional[Any] = None) -> None:
        self._model = model

    @classmethod
    def is_available(cls) -> bool:
        # As with the CTAB-GAN TF backend: no eager TF import.  Real
        # absence of TensorFlow surfaces as ImportError when fit/load
        # tries to use it.
        return True

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: Any,
        labels: Any,
        categorical_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Train DFCGAN on (data, labels).

        Args:
            data:    np.ndarray of shape (N, seq_len, num_features)
            labels:  one-hot np.ndarray of shape (N, num_classes)
            **kwargs: hyperparameters (latent_dim, d_steps,
                     instance_noise_std, label_smoothing, fm_weight,
                     fm_var_weight, mmd_weight, generator_arch,
                     gen_base_filters, gen_kernel_size,
                     gen_upsample_blocks, batch_size, epochs,
                     steps_per_epoch, learning_rate, verbose).  Unknown
                     keys are silently ignored — see GANBackend docs.
        """
        import tensorflow as tf  # noqa: E402
        from GANs.df_cgan import DFCGAN  # noqa: E402

        data_f32 = np.asarray(data, dtype="float32")
        labels_f32 = np.asarray(labels, dtype="float32")
        if data_f32.ndim != 3:
            raise ValueError(
                f"CGAN requires 3D input (N, seq_len, features), "
                f"got shape {data_f32.shape}"
            )
        num_samples, seq_len, num_features = data_f32.shape
        num_classes = labels_f32.shape[1]

        # Per-feature statistics — used by DFCGAN for output
        # rescaling/clipping inside the generator.
        feature_mean = data_f32.mean(axis=(0, 1)).astype("float32")
        feature_std = data_f32.std(axis=(0, 1)).astype("float32")
        feature_min = data_f32.min(axis=(0, 1)).astype("float32")
        feature_max = data_f32.max(axis=(0, 1)).astype("float32")

        batch_size = int(kwargs.get("batch_size", 256))
        learning_rate = float(kwargs.get("learning_rate", 3e-4))

        ds = (
            tf.data.Dataset.from_tensor_slices((data_f32, labels_f32))
            .shuffle(buffer_size=min(8192, num_samples))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        gan = DFCGAN(
            seq_len=seq_len,
            num_features=num_features,
            num_classes=num_classes,
            latent_dim=int(kwargs.get("latent_dim", 64)),
            d_steps=int(kwargs.get("d_steps", 2)),
            instance_noise_std=float(kwargs.get("instance_noise_std", 0.01)),
            label_smoothing=bool(kwargs.get("label_smoothing", True)),
            fm_weight=float(kwargs.get("fm_weight", 1.0)),
            fm_var_weight=float(kwargs.get("fm_var_weight", 0.5)),
            mmd_weight=float(kwargs.get("mmd_weight", 0.5)),
            feature_mean=feature_mean,
            feature_std=feature_std,
            feature_min=feature_min,
            feature_max=feature_max,
            generator_arch=str(kwargs.get("generator_arch", "cnn")),
            gen_base_filters=int(kwargs.get("gen_base_filters", 128)),
            gen_kernel_size=int(kwargs.get("gen_kernel_size", 3)),
            gen_upsample_blocks=int(kwargs.get("gen_upsample_blocks", 2)),
        )
        gan.compile(
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        )

        epochs = int(kwargs.get("epochs", 100))
        steps_per_epoch = kwargs.get("steps_per_epoch")
        verbose = 1 if kwargs.get("verbose", True) else 0

        gan.fit(ds, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=verbose)

        # Stash the metadata needed to reconstruct the model in
        # ``_load_cgan_model``.  Mirrors the dict GANInterface used to
        # build inline.
        gan._interface_metadata = {
            "seq_len": seq_len,
            "num_features": num_features,
            "num_classes": num_classes,
            "latent_dim": gan.latent_dim,
            "d_steps": gan.d_steps,
            "instance_noise_std": gan.instance_noise_std,
            "label_smoothing": gan.label_smoothing,
            "fm_weight": gan.fm_weight,
            "fm_var_weight": gan.fm_var_weight,
            "mmd_weight": gan.mmd_weight,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_min": feature_min,
            "feature_max": feature_max,
            "generator_arch": kwargs.get("generator_arch", "cnn"),
            "gen_base_filters": int(kwargs.get("gen_base_filters", 128)),
            "gen_kernel_size": int(kwargs.get("gen_kernel_size", 3)),
            "gen_upsample_blocks": int(kwargs.get("gen_upsample_blocks", 2)),
            "learning_rate": learning_rate,
        }
        self._model = gan

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    def generate(self, n: int, **kwargs: Any) -> Any:
        if self._model is None:
            raise RuntimeError(
                "CGANTFBackend.generate called before fit/load — no model"
            )
        one_hot = kwargs.get("one_hot")
        if one_hot is None:
            raise ValueError(
                "generate() for CGAN requires keyword argument one_hot=<np.ndarray>"
            )
        return self._model.generate(n, one_hot)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, save_path: str, **extra_metadata: Any) -> None:
        if self._model is None:
            raise RuntimeError("CGANTFBackend.save called before fit — no model")

        from GANs.df_cgan import _save_cgan_model  # noqa: E402

        meta = dict(getattr(self._model, "_interface_metadata", {}))
        meta.update(extra_metadata)
        _save_cgan_model(self._model, meta, save_path)

    @classmethod
    def load(cls, save_path: str) -> Tuple["CGANTFBackend", Dict[str, Any]]:
        from GANs.df_cgan import _load_cgan_model  # noqa: E402

        gan, metadata = _load_cgan_model(save_path)
        if gan is None:
            raise FileNotFoundError(
                f"No saved CGAN model found at {save_path}"
            )
        instance = cls()
        instance._model = gan
        # Mirror what GANInterface used to do inline so any consumer
        # reading _interface_metadata still finds it.
        instance._model._interface_metadata = metadata or {}
        return instance, metadata or {}
