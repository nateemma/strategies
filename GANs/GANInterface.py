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

from typing import Any, Dict, List, Optional

from GANs.GANType import GANType
from GANs.GANBackend import (
    GANBackend,
    fit_with_fallback,
    load_with_fallback,
)
import GANs.backends  # noqa: F401  — side-effect: registers concrete backends


# ---------------------------------------------------------------------------
# Phase 2 migration tracker
# ---------------------------------------------------------------------------

# GAN types that have been migrated to the GANBackend registry.  Other
# types still go through the type-specific dispatch methods below.  As
# more types migrate (CGAN, WGAN, MT_WGAN), they'll be added to this set
# and the corresponding _fit_*/load branches removed.
_BACKEND_MIGRATED: set = {
    GANType.CTAB_GAN,
    GANType.MT_CTAB_GAN,
    GANType.CGAN,
    GANType.WGAN,
    GANType.MT_WGAN,
}


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
        # Legacy slot for non-migrated GAN types.  Migrated types use
        # ``self._backend`` (a GANBackend instance) instead.  See
        # ``_BACKEND_MIGRATED`` at module top.
        self._model: Optional[Any] = None
        self._backend: Optional[GANBackend] = None

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
        # Phase 2: types migrated to the GANBackend registry go through
        # resolve_backend → backend.fit, fully bypassing the
        # type-specific _fit_* methods below.  The resolved config is
        # passed as **kwargs and the backend silently drops keys it
        # doesn't recognise.
        # Phase 2 complete: every GANType with fit() support goes through
        # the registry.  fit_with_fallback handles MLX→TF graceful
        # fallback when an MLX backend's underlying module fails to
        # import.  Backends silently ignore kwargs they don't recognise.
        if self.gan_type not in _BACKEND_MIGRATED:
            raise ValueError(
                f"fit() is not supported for GANType.{self.gan_type.name}."
            )
        config = self._resolved_config(**caller_overrides)
        # Drop save_path — it's a separate save() call, not a fit kwarg.
        config.pop("save_path", None)
        self._backend = fit_with_fallback(
            self.gan_type, data, labels,
            prefer_mlx=self._use_mlx,
            categorical_columns=categorical_columns,
            **config,
        )
        # Mirror onto self._model for any caller still inspecting it.
        self._model = getattr(self._backend, "_model", None)

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
        # Phase 2: migrated types delegate straight to the backend.
        if self._backend is not None:
            return self._backend.generate(n, **kwargs)

        if self._model is None:
            raise RuntimeError(
                "No model is available.  Call fit() to train one or "
                "load() to restore a saved model."
            )
        # All migrated types delegate via the early return at the top
        # of this method.  Reaching here means the legacy mock-based
        # tests set self._model directly without going through fit/load;
        # keep legacy delegation + arg validation for that case.
        if self.gan_type in (GANType.WGAN, GANType.CGAN):
            one_hot = kwargs.get("one_hot")
            if one_hot is None:
                raise ValueError(
                    f"generate() for {self.gan_type.name} requires "
                    f"keyword argument one_hot=<np.ndarray>"
                )
            return self._model.generate(n, one_hot)

        if self.gan_type == GANType.MT_WGAN:
            task_labels = kwargs.get("task_labels")
            if task_labels is None:
                raise ValueError(
                    "generate() for MT_WGAN requires keyword argument task_labels=<dict>"
                )
            return self._model.generate(n, task_labels)

        # CTAB-GAN family — legacy mock-test fallback
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
        # Phase 2: migrated types delegate straight to the backend.
        if self._backend is not None:
            if self.save_path is None:
                raise RuntimeError("save_path is None; cannot persist model.")
            self._backend.save(self.save_path, **extra_metadata)
            return

        if self._model is None:
            raise RuntimeError("No model to save.  Call fit() first.")
        if self.save_path is None:
            raise RuntimeError("save_path is None; cannot persist model.")

        # Migrated types reach here only when the caller bypassed fit()
        # and set ``self._model`` directly (the existing mock-based tests
        # in test_gan_interface.py do this).  Delegate to model.save
        # using the same one-line shape the legacy code did.
        if self.gan_type in _BACKEND_MIGRATED:
            self._model.save(self.save_path, **extra_metadata)
            return

        # Every supported GANType is handled via the backend registry —
        # see the early return at the top of save().  Reaching here means
        # an unhandled GANType slipped through.
        raise ValueError(
            f"save() not supported for GANType.{self.gan_type.name}"
        )

    def load(self) -> Dict[str, Any]:
        """Restore a saved model from save_path.

        For CTAB_GAN, automatically selects the MLX variant when its
        metadata file is present and MLX is available.

        Returns:
            Metadata dict stored alongside the model (e.g. thresholds).
        """
        # Phase 2: migrated types use the GANBackend registry's
        # MLX-then-TF fallback probe.
        if self.gan_type in _BACKEND_MIGRATED:
            backend, metadata = load_with_fallback(
                self.gan_type, self.save_path, prefer_mlx=self._use_mlx
            )
            self._backend = backend
            self._model = getattr(backend, "_model", None)
            return metadata or {}

        # Every supported GANType is handled by the backend registry —
        # see the early return at the top of load().  Reaching here
        # means an unhandled GANType slipped through.
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

    # Phase 2 cleanup: every per-type ``_fit_*`` helper, the
    # ``_build_model`` factory, and the CTAB constructor-key constants
    # have been removed.  Each backend now owns its own kwarg
    # partitioning and lazy module imports — see GANs/backends/*.py.
