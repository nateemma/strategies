# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore
"""
GANInterface — Unified interface for all GAN implementations.

All GAN types use the explicit lifecycle: fit() / generate() / save() / load().

Usage — WGAN (tabular, 2D input):
    interface = GANInterface(GANType.WGAN, save_path="/path/to/GANs")
    interface.fit(data_2d, labels_1hot)
    interface.save()
    gen_data = interface.generate(n=50, one_hot=np.eye(3)[1:2].repeat(50, axis=0))

Usage — MT_WGAN (multi-task, 3D input):
    interface = GANInterface(GANType.MT_WGAN, save_path="/path/to/GANs")
    interface.fit(data_3d, {"trading": trading_oh, "regime": regime_oh})
    interface.save()
    gen_data, gen_labels = interface.generate(
        n=50, task_labels={"trading": trading_oh_50, "regime": regime_oh_50}
    )

Usage — CTAB_GAN (DataFrame input):
    interface = GANInterface(GANType.CTAB_GAN, save_path="/path/to/CTABGANs")
    interface.fit(train_df, labels_one_hot, categorical_columns=[...])
    interface.save(min_buy_gain_threshold=0.016)
    thresholds = interface.load()
    generated_df = interface.generate(num_samples=500, class_label=1)

Usage — CGAN (sequential, 3D input):
    interface = GANInterface(GANType.CGAN, save_path="/path/to/CGANs")
    interface.fit(data_3d, labels_1hot)   # data_3d: (N, seq_len, features)
    interface.save()
    gen_data = interface.generate(n=50, one_hot=labels_1hot[:50])
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from GANs.GANType import GANType


# ---------------------------------------------------------------------------
# MLX detection — once at import time
# ---------------------------------------------------------------------------

def _detect_mlx() -> bool:
    try:
        import mlx.core as mx  # type: ignore
        return hasattr(mx, "metal") and mx.metal.is_available()
    except (ImportError, ModuleNotFoundError):
        return False


_HAS_MLX: bool = _detect_mlx()


# ---------------------------------------------------------------------------
# GANInterface
# ---------------------------------------------------------------------------

class GANInterface:
    """Unified, backend-agnostic interface for all GAN types.

    GAN-internal training parameters (epochs, batch size, critic steps,
    latent dim, …) are stored as per-type defaults inside this class and
    are intentionally not exposed to callers.

    Callers only need to express:
      - *which* GAN to use (GANType enum)
      - *where* to read/write model files (save_path)
      - *strategy intent*: how much augmentation is wanted
        (augmentation_target_ratio, task_target_ratios, seq_len)

    Args:
        gan_type:   Which backend to use (see GANType).
        save_path:  Directory for model files (reading and writing).
        prefer_mlx: Use the MLX backend when available (default True).
    """

    # ---------------------------------------------------------------------- #
    # Per-type training defaults — hidden from callers.                      #
    # Callers may pass overrides to fit() for any of these keys.            #
    # ---------------------------------------------------------------------- #
    _DEFAULTS: Dict[GANType, Dict[str, Any]] = {
        GANType.WGAN: {
            "epochs":     100,
            "batch_size": 2048,
            "n_critic":   5,
            "noise_std":  0.02,
            "verbose":    True,
        },
        GANType.MT_WGAN: {
            "epochs":     100,
            "batch_size": 2048,
            "n_critic":   6,
            "verbose":    True,
            "seq_len":    1,
        },
        GANType.CTAB_GAN: {
            "epochs":                300,
            "batch_size":            2048,
            "latent_dim":            128,
            "generator_layers":      [256, 256],
            "discriminator_layers":  [256, 256],
            "learning_rate":         2e-4,
            "beta_1":                0.2,
            "beta_2":                0.999,
            "gp_weight":             10.0,
            "verbose":               True,
        },
        GANType.MT_CTAB_GAN: {
            "epochs":     300,
            "batch_size": 2048,
            "latent_dim": 128,
            "verbose":    True,
        },
        GANType.CGAN: {
            "epochs":            100,
            "batch_size":        256,
            "d_steps":           2,
            "steps_per_epoch":   None,
            "learning_rate":     3e-4,
            "instance_noise_std": 0.01,
            "label_smoothing":   True,
            "fm_weight":         1.0,
            "fm_var_weight":     0.5,
            "mmd_weight":        0.5,
            "generator_arch":    "cnn",
            "gen_base_filters":  128,
            "gen_kernel_size":   3,
            "gen_upsample_blocks": 2,
            "verbose":           True,
        },
    }

    def __init__(
        self,
        gan_type: GANType,
        save_path: str,
        prefer_mlx: bool = True,
    ) -> None:
        self.gan_type = gan_type
        self.save_path = save_path
        self._use_mlx: bool = prefer_mlx and _HAS_MLX
        self._model: Optional[Any] = None

    # ---------------------------------------------------------------------- #
    # fit() / generate() / save() / load() — all GAN types                  #
    # ---------------------------------------------------------------------- #

    def fit(
        self,
        data: Any,
        labels: Any,
        categorical_columns: Optional[List[str]] = None,
        **caller_overrides: Any,
    ) -> None:
        """Train a GAN model.

        Supported for all GAN types (WGAN, MT_WGAN, CTAB_GAN, MT_CTAB_GAN,
        CGAN).  After fitting, call generate() for inference and save() to
        persist.

        Args:
            data:               Training data.  numpy array (N, F) for WGAN;
                                (N, seq_len, F) for MT_WGAN and CGAN;
                                DataFrame for CTAB-GAN variants.
            labels:             One-hot array (N, C) for single-task GANs,
                                or a dict {task: one_hot} for multi-task.
            categorical_columns: CTAB-GAN only — columns to treat as
                                 categorical (auto-detected when None).
            **caller_overrides: Override any default training parameter.
        """
        _FIT_DISPATCH = {
            GANType.WGAN:        self._fit_wgan,
            GANType.MT_WGAN:     self._fit_mt_wgan,
            GANType.CTAB_GAN:    self._fit_ctab,
            GANType.MT_CTAB_GAN: self._fit_ctab,
            GANType.CGAN:        self._fit_cgan,
        }
        if self.gan_type not in _FIT_DISPATCH:
            raise ValueError(
                f"fit() is not supported for GANType.{self.gan_type.name}."
            )
        _FIT_DISPATCH[self.gan_type](data, labels, categorical_columns, caller_overrides)

    def generate(self, n: int, **kwargs: Any) -> Any:
        """Generate synthetic samples from a fitted or loaded model.

        For WGAN:
            generate(n, one_hot=<np.ndarray (n, C)>)
            Returns np.ndarray (n, 1, F) — 3D with seq_len=1.

        For MT_WGAN:
            generate(n, task_labels=<dict[str, np.ndarray]>)
            Returns (np.ndarray (n, 1, F), dict[str, np.ndarray]).

        For CGAN:
            generate(n, one_hot=<np.ndarray (n, C)>)
            Returns np.ndarray (n, seq_len, F).

        For CTAB_GAN / MT_CTAB_GAN:
            generate(n, class_label=<int>)  or  generate(n, task_labels=<dict>)
            Returns pd.DataFrame.

        Raises:
            RuntimeError: If neither fit() nor load() has been called.
        """
        if self._model is None:
            raise RuntimeError(
                "No model is available.  Call fit() to train one or "
                "load() to restore a saved model."
            )
        if self.gan_type == GANType.WGAN:
            one_hot = kwargs.get("one_hot")
            if one_hot is None:
                raise ValueError("generate() for WGAN requires keyword argument one_hot=<np.ndarray>")
            return self._model.generate(n, one_hot)

        if self.gan_type == GANType.MT_WGAN:
            task_labels = kwargs.get("task_labels")
            if task_labels is None:
                raise ValueError("generate() for MT_WGAN requires keyword argument task_labels=<dict>")
            return self._model.generate(n, task_labels)

        if self.gan_type == GANType.CGAN:
            one_hot = kwargs.get("one_hot")
            if one_hot is None:
                raise ValueError("generate() for CGAN requires keyword argument one_hot=<np.ndarray>")
            return self._model.generate(n, one_hot)

        # CTAB-GAN family — delegate unchanged
        num_samples = kwargs.pop("num_samples", n)
        return self._model.generate(num_samples=num_samples, **kwargs)

    def save(self, **extra_metadata: Any) -> None:
        """Persist a fitted model to save_path.

        For WGAN / MT_WGAN, extra_metadata is merged into the training
        metadata (e.g. min_buy_gain_threshold, training_type).

        Raises:
            RuntimeError: If neither fit() nor load() has been called, or
                          save_path is None.
        """
        if self._model is None:
            raise RuntimeError("No model to save.  Call fit() first.")
        if self.save_path is None:
            raise RuntimeError("save_path is None; cannot persist model.")

        if self.gan_type == GANType.WGAN:
            # MLX path: WGANMLX has its own save() for weights; we write
            # wgangp_metadata.pkl alongside it for load() to reconstruct.
            try:
                from GANs.df_wgan_mlx import WGANMLX  # noqa: E402
                if isinstance(self._model, WGANMLX):
                    import pickle  # noqa: E402
                    os.makedirs(self.save_path, exist_ok=True)
                    self._model.save(self.save_path)  # → wgan_gen_mlx.safetensors
                    meta = {
                        "num_features": self._model.num_features,
                        "num_classes":  self._model.num_classes,
                        "latent_dim":   self._model.latent_dim,
                        "seq_len":      1,
                    }
                    meta.update(extra_metadata)
                    with open(os.path.join(self.save_path, "wgangp_metadata.pkl"), "wb") as _f:
                        pickle.dump(meta, _f)
                    return
            except (ImportError, ModuleNotFoundError):
                pass
            # TF path
            from GANs.df_wgan_gp import _save_wgan_model  # noqa: E402
            meta = dict(getattr(self._model, "_interface_metadata", {}))
            meta.update(extra_metadata)
            _save_wgan_model(self._model, meta, self.save_path)
            return

        if self.gan_type == GANType.MT_WGAN:
            # MLX path: MTWGANMLX saves its own weights + pickle metadata.
            try:
                from GANs.df_mt_wgan_mlx import MTWGANMLX  # noqa: E402
                if isinstance(self._model, MTWGANMLX):
                    meta = {
                        "seq_len":         self._model.seq_len,
                        "num_features":    self._model.num_features,
                        "task_label_dims": dict(self._model.task_label_dims),
                        "latent_dim":      self._model.latent_dim,
                    }
                    meta.update(extra_metadata)
                    self._model.save(self.save_path, meta)
                    return
            except (ImportError, ModuleNotFoundError):
                pass
            # TF path — sanitise TrackedDict before pickling.
            from GANs.df_mt_wgan_gp import _save_mt_wgan_model  # noqa: E402
            meta = dict(getattr(self._model, "_interface_metadata", {}))
            # Keras wraps dict attributes on Model subclasses as TrackedDict,
            # which is not picklable.  Convert those entries to plain Python
            # dicts/floats before handing off to the pickle-based serialiser.
            if "task_label_dims" in meta:
                meta["task_label_dims"] = dict(meta["task_label_dims"])
            if "task_loss_weights" in meta:
                meta["task_loss_weights"] = {
                    k: float(v) for k, v in meta["task_loss_weights"].items()
                }
            meta.update(extra_metadata)
            _save_mt_wgan_model(self._model, meta, self.save_path)
            return

        if self.gan_type == GANType.CGAN:
            from GANs.df_cgan import _save_cgan_model  # noqa: E402
            meta = dict(getattr(self._model, "_interface_metadata", {}))
            meta.update(extra_metadata)
            _save_cgan_model(self._model, meta, self.save_path)
            return

        # CTAB-GAN family
        self._model.save(self.save_path, **extra_metadata)

    def load(self) -> Dict[str, Any]:
        """Restore a saved model from save_path.

        For CTAB_GAN, automatically selects the MLX variant when its
        metadata file is present and MLX is available.

        Returns:
            Metadata dict stored alongside the model (e.g. thresholds).
        """
        if self.gan_type == GANType.WGAN:
            # MLX path: wgan_gen_mlx.safetensors present + prefer_mlx
            if self._use_mlx:
                mlx_weights = os.path.join(self.save_path, "wgan_gen_mlx.safetensors")
                mlx_meta    = os.path.join(self.save_path, "wgangp_metadata.pkl")
                if os.path.exists(mlx_weights) and os.path.exists(mlx_meta):
                    try:
                        import pickle  # noqa: E402
                        from GANs.df_wgan_mlx import WGANMLX  # noqa: E402
                        with open(mlx_meta, "rb") as _f:
                            metadata = pickle.load(_f)
                        gan = WGANMLX(
                            metadata["num_features"],
                            metadata["num_classes"],
                            latent_dim=metadata.get("latent_dim", 64),
                        )
                        gan.load(self.save_path)
                        self._model = gan
                        return metadata
                    except (ImportError, ModuleNotFoundError):
                        pass
            # TF path
            from GANs.df_wgan_gp import _load_wgan_model  # noqa: E402
            gan, metadata = _load_wgan_model(self.save_path)
            if gan is None:
                raise RuntimeError(
                    f"No saved WGAN model found at {self.save_path}"
                )
            self._model = gan
            self._model._interface_metadata = metadata or {}
            return metadata or {}

        if self.gan_type == GANType.MT_WGAN:
            # MLX path.  MTWGANMLX writes its files with a ``_s{seq_len}``
            # suffix when seq_len > 1 (e.g. mt_wgan_meta_mlx_s16.pkl),
            # so we glob for the metadata file rather than probing a
            # single fixed name.
            if self._use_mlx:
                import glob  # noqa: E402
                meta_candidates = sorted(glob.glob(
                    os.path.join(self.save_path, "mt_wgan_meta_mlx*.pkl")
                ))
                if meta_candidates:
                    mlx_meta = meta_candidates[0]
                    if len(meta_candidates) > 1:
                        print(
                            f"    Multiple MT_WGAN MLX metadata files found in "
                            f"{self.save_path}: {meta_candidates}.  Using {mlx_meta}."
                        )
                    try:
                        import pickle  # noqa: E402
                        from GANs.df_mt_wgan_mlx import MTWGANMLX  # noqa: E402
                        with open(mlx_meta, "rb") as _f:
                            metadata = pickle.load(_f)
                        gan = MTWGANMLX(
                            seq_len=metadata["seq_len"],
                            num_features=metadata["num_features"],
                            task_label_dims=metadata["task_label_dims"],
                            latent_dim=metadata.get("latent_dim", 64),
                        )
                        # MTWGANMLX.load() reconstructs the per-seq_len paths
                        # internally via _get_paths, so it finds the matching
                        # weight files even when the suffix is present.
                        ok, _ = gan.load(self.save_path)
                        if not ok:
                            raise RuntimeError(
                                f"MTWGANMLX.load() failed for {self.save_path} "
                                f"(metadata file: {mlx_meta})"
                            )
                        self._model = gan
                        return metadata
                    except (ImportError, ModuleNotFoundError):
                        pass
            # TF path
            from GANs.df_mt_wgan_gp import _load_mt_wgan_model  # noqa: E402
            gan, metadata = _load_mt_wgan_model(self.save_path)
            if gan is None:
                raise RuntimeError(
                    f"No saved MT_WGAN model found at {self.save_path}"
                )
            self._model = gan
            self._model._interface_metadata = metadata or {}
            return metadata or {}

        if self.gan_type == GANType.CTAB_GAN:
            mlx_meta = os.path.join(self.save_path, "metadata_mlx.pkl")
            if self._use_mlx and os.path.exists(mlx_meta):
                try:
                    from GANs.df_ctab_gan_mlx import CTABGANMLX  # noqa: E402
                    self._model = CTABGANMLX()
                    return self._model.load(self.save_path)
                except (ImportError, ModuleNotFoundError):
                    pass
            from GANs.df_ctab_gan import CTABGANPlus  # noqa: E402
            self._model = CTABGANPlus()
            return self._model.load(self.save_path)

        if self.gan_type == GANType.MT_CTAB_GAN:
            from GANs.df_mt_ctab_gan import CTABGANPlusMT  # noqa: E402
            self._model = CTABGANPlusMT()
            return self._model.load(self.save_path)

        if self.gan_type == GANType.CGAN:
            from GANs.df_cgan import _load_cgan_model  # noqa: E402
            gan, metadata = _load_cgan_model(self.save_path)
            if gan is None:
                raise RuntimeError(
                    f"No saved CGAN model found at {self.save_path}"
                )
            self._model = gan
            self._model._interface_metadata = metadata or {}
            return metadata or {}

        raise ValueError(
            f"load() is not supported for GANType.{self.gan_type.name}."
        )

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                        #
    # ---------------------------------------------------------------------- #

    def _resolved_config(self, **caller_overrides: Any) -> Dict[str, Any]:
        """Merge type defaults with caller overrides, injecting save_path."""
        config = dict(self._DEFAULTS.get(self.gan_type, {}))
        config.update(caller_overrides)
        config["save_path"] = self.save_path
        return config

    # Keys accepted by the CTABGANPlus / CTABGANPlusMT constructors.
    # Anything in _resolved_config that is NOT in this set and NOT in
    # _CTAB_SKIP_KEYS will be forwarded to model.fit() (typically only
    # validation_split).
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

    # Keys that belong to other GAN types or callers; silently ignored for CTAB.
    _CTAB_SKIP_KEYS: frozenset = frozenset({
        "save_path", "augmentation_target_ratio", "task_target_ratios",
        "assess_quality", "n_critic", "noise_std", "architecture", "seq_len",
    })

    # Keys accepted by the lightweight CTABGANMLX constructor.
    _CTAB_MLX_CTOR_KEYS: frozenset = frozenset({"latent_dim", "epochs", "batch_size", "n_critic", "verbose"})

    def _build_model(self, **ctor_kwargs: Any) -> Any:
        """Instantiate the correct model class for model-based GANs.

        Args:
            **ctor_kwargs: Constructor parameters forwarded to the model class
                           (e.g. epochs, batch_size, latent_dim).  Unknown keys
                           are silently dropped for the MLX variant (which has a
                           smaller constructor surface).
        """
        if self.gan_type == GANType.CTAB_GAN:
            if self._use_mlx:
                try:
                    from GANs.df_ctab_gan_mlx import CTABGANMLX  # noqa: E402
                    mlx_kw = {k: v for k, v in ctor_kwargs.items()
                               if k in self._CTAB_MLX_CTOR_KEYS}
                    return CTABGANMLX(**mlx_kw)
                except (ImportError, ModuleNotFoundError):
                    pass
            from GANs.df_ctab_gan import CTABGANPlus  # noqa: E402
            return CTABGANPlus(**ctor_kwargs)

        if self.gan_type == GANType.MT_CTAB_GAN:
            from GANs.df_mt_ctab_gan import CTABGANPlusMT  # noqa: E402
            return CTABGANPlusMT(**ctor_kwargs)

        raise ValueError(
            f"_build_model() called for non-model-based type: {self.gan_type.name}"
        )

    def _fit_wgan(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        _categorical_columns: Optional[List[str]],
        caller_overrides: Dict[str, Any],
    ) -> None:
        data_f32 = np.asarray(data, dtype="float32")
        labels_f32 = np.asarray(labels, dtype="float32")
        if data_f32.ndim == 3:
            # Caller passed 3D; squeeze to 2D for balance_with_wgan_gp
            data_f32 = data_f32[:, 0, :]
        config = self._resolved_config(**caller_overrides)
        config.setdefault("seq_len", 1)
        config["save_path"] = None              # caller must call save() explicitly
        config["assess_quality"] = False        # assessed separately if needed
        config["augmentation_target_ratio"] = 0.0   # train only; no generation needed
        if self._use_mlx:
            try:
                from GANs.df_wgan_mlx import balance_with_wgan_mlx  # noqa: E402
                _, _, gan = balance_with_wgan_mlx(
                    data_f32, labels_f32, _return_model=True, **config
                )
                self._model = gan
                return
            except (ImportError, ModuleNotFoundError):
                pass
        from GANs.df_wgan_gp import balance_with_wgan_gp  # noqa: E402
        _, _, gan = balance_with_wgan_gp(
            data_f32, labels_f32, _return_model=True, **config
        )
        self._model = gan

    def _fit_mt_wgan(
        self,
        data: np.ndarray,
        labels: Dict[str, np.ndarray],
        _categorical_columns: Optional[List[str]],
        caller_overrides: Dict[str, Any],
    ) -> None:
        data_f32 = np.asarray(data, dtype="float32")
        config = self._resolved_config(**caller_overrides)
        config["save_path"] = None
        config["assess_quality"] = False
        # Always skip the generation loop — fit() trains only; generate() handles synthesis
        config["task_target_ratios"] = {k: 0.0 for k in labels}
        if self._use_mlx:
            try:
                from GANs.df_mt_wgan_mlx import balance_with_mt_wgan_mlx  # noqa: E402
                _, _, gan = balance_with_mt_wgan_mlx(
                    data_f32, labels, _return_model=True, **config
                )
                self._model = gan
                return
            except (ImportError, ModuleNotFoundError):
                pass
        from GANs.df_mt_wgan_gp import balance_with_mt_wgan_gp  # noqa: E402
        _, _, gan = balance_with_mt_wgan_gp(
            data_f32, labels, _return_model=True, **config
        )
        self._model = gan

    def _fit_ctab(
        self,
        data: pd.DataFrame,
        labels: Any,
        categorical_columns: Optional[List[str]],
        caller_overrides: Dict[str, Any],
    ) -> None:
        config = self._resolved_config(**caller_overrides)
        # Route training hyperparams to the constructor (CTABGANPlus/MT takes
        # epochs, batch_size, etc. at construction time, not at fit() time).
        ctor_kwargs = {k: v for k, v in config.items() if k in self._CTAB_CTOR_KEYS}
        # Anything not going to the constructor and not GAN-internal goes to fit()
        # (in practice this is usually just validation_split).
        fit_kwargs = {
            k: v for k, v in config.items()
            if k not in self._CTAB_CTOR_KEYS and k not in self._CTAB_SKIP_KEYS
        }
        self._model = self._build_model(**ctor_kwargs)
        self._model.fit(
            dataframe=data,
            labels=labels,
            categorical_columns=categorical_columns or [],
            **fit_kwargs,
        )

    def _fit_cgan(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        _categorical_columns: Optional[List[str]],
        caller_overrides: Dict[str, Any],
    ) -> None:
        import tensorflow as tf
        from GANs.df_cgan import DFCGAN  # noqa: E402

        data_f32 = np.asarray(data, dtype="float32")
        labels_f32 = np.asarray(labels, dtype="float32")
        if data_f32.ndim != 3:
            raise ValueError(
                f"CGAN requires 3D input (N, seq_len, features), got shape {data_f32.shape}"
            )
        num_samples, seq_len, num_features = data_f32.shape
        num_classes = labels_f32.shape[1]

        config = self._resolved_config(**caller_overrides)

        # Feature stats used by the generator for output rescaling/clipping
        feature_mean = data_f32.mean(axis=(0, 1)).astype("float32")
        feature_std  = data_f32.std(axis=(0, 1)).astype("float32")
        feature_min  = data_f32.min(axis=(0, 1)).astype("float32")
        feature_max  = data_f32.max(axis=(0, 1)).astype("float32")

        batch_size    = int(config.get("batch_size", 256))
        learning_rate = float(config.get("learning_rate", 3e-4))

        ds = tf.data.Dataset.from_tensor_slices((data_f32, labels_f32))
        ds = ds.shuffle(buffer_size=min(8192, num_samples)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        gan = DFCGAN(
            seq_len=seq_len,
            num_features=num_features,
            num_classes=num_classes,
            latent_dim=int(config.get("latent_dim", 64)),
            d_steps=int(config.get("d_steps", 2)),
            instance_noise_std=float(config.get("instance_noise_std", 0.01)),
            label_smoothing=bool(config.get("label_smoothing", True)),
            fm_weight=float(config.get("fm_weight", 1.0)),
            fm_var_weight=float(config.get("fm_var_weight", 0.5)),
            mmd_weight=float(config.get("mmd_weight", 0.5)),
            feature_mean=feature_mean,
            feature_std=feature_std,
            feature_min=feature_min,
            feature_max=feature_max,
            generator_arch=str(config.get("generator_arch", "cnn")),
            gen_base_filters=int(config.get("gen_base_filters", 128)),
            gen_kernel_size=int(config.get("gen_kernel_size", 3)),
            gen_upsample_blocks=int(config.get("gen_upsample_blocks", 2)),
        )
        gan.compile(
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        )

        epochs           = int(config.get("epochs", 100))
        steps_per_epoch  = config.get("steps_per_epoch")
        verbose          = 1 if config.get("verbose", True) else 0

        gan.fit(ds, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=verbose)

        # Metadata needed to reconstruct the model via _load_cgan_model
        gan._interface_metadata = {
            "seq_len":             seq_len,
            "num_features":        num_features,
            "num_classes":         num_classes,
            "latent_dim":          gan.latent_dim,
            "d_steps":             gan.d_steps,
            "instance_noise_std":  gan.instance_noise_std,
            "label_smoothing":     gan.label_smoothing,
            "fm_weight":           gan.fm_weight,
            "fm_var_weight":       gan.fm_var_weight,
            "mmd_weight":          gan.mmd_weight,
            "feature_mean":        feature_mean,
            "feature_std":         feature_std,
            "feature_min":         feature_min,
            "feature_max":         feature_max,
            "generator_arch":      config.get("generator_arch", "cnn"),
            "gen_base_filters":    int(config.get("gen_base_filters", 128)),
            "gen_kernel_size":     int(config.get("gen_kernel_size", 3)),
            "gen_upsample_blocks": int(config.get("gen_upsample_blocks", 2)),
            "learning_rate":       learning_rate,
        }
        self._model = gan
