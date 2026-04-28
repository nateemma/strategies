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

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from GANs.GANType import GANType
from GANs.GANBackend import (
    GANBackend,
    fit_with_fallback,
    load_with_fallback,
)
import GANs.backends  # noqa: F401  — side-effect: registers concrete backends


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class GANMetadataMismatchError(ValueError):
    """Raised by GANInterface.load(expected=...) when persisted metadata
    drifts from what the strategy declared.

    A mismatch means either the GAN must be retrained or the strategy
    must be updated to match — silently using stale data would corrupt
    training (e.g. labels generated under a different threshold than the
    GAN saw).  We refuse to continue and surface the diff to the caller.
    """

    def __init__(self, mismatches: Dict[str, Any], gan_type: GANType, save_path: str):
        self.mismatches = mismatches
        self.gan_type = gan_type
        self.save_path = save_path
        lines = [
            f"GAN metadata mismatch for {gan_type.name} at {save_path}:",
        ]
        for key, (saved, expected) in mismatches.items():
            lines.append(
                f"  {key}: saved={saved!r}  expected={expected!r}"
            )
        lines.append(
            "  Either retrain the GAN with the current strategy config, "
            "or update the strategy to match what was saved."
        )
        super().__init__("\n".join(lines))


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def _assert_generated_finite(result: Any, *, gan_type: GANType) -> None:
    """
    Raise ValueError if ``result`` (the return value of generate())
    contains any non-finite values.

    Handles the four output shapes the various backends produce:
      * np.ndarray (WGAN, CGAN)
      * tuple of (samples_ndarray, labels_dict) (MT_WGAN)
      * pd.DataFrame (CTAB_GAN)
      * tuple of (DataFrame, labels_dict) (MT_CTAB_GAN)

    The labels portion of the multi-task tuples is always one-hot
    finite by construction; we don't bother checking it.
    """
    samples = result[0] if isinstance(result, tuple) else result

    # DataFrames need .values; everything else is array-like.
    if hasattr(samples, "values") and not isinstance(samples, np.ndarray):
        arr = np.asarray(samples.values)
    else:
        arr = np.asarray(samples)

    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(
            f"generate() for GANType.{gan_type.name} produced "
            f"{n_bad} non-finite values (out of {arr.size}).  This "
            f"usually means the generator's output isn't bounded "
            f"(WGAN-MLX in particular has no _postprocess clip "
            f"today).  Letting these reach a classifier corrupts "
            f"training — fix the generator or pre-clip the output."
        )


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
            ValueError:   If the generated output contains NaN/Inf
                          values.  Generators that haven't fully
                          converged or that lack output clipping
                          (see WGAN-MLX) can produce non-finite
                          samples; we reject them at the boundary
                          rather than let them poison downstream
                          consumers (the failure mode that
                          surfaced in NNMT_WGAN_MLX training).
        """
        result = self._dispatch_generate(n, **kwargs)
        _assert_generated_finite(result, gan_type=self.gan_type)
        return result

    def _dispatch_generate(self, n: int, **kwargs: Any) -> Any:
        """Internal: do the actual generate without the finiteness check."""
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

    def load(
        self,
        expected: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Restore a saved model from save_path and (optionally) validate
        its persisted metadata against caller expectations.

        For CTAB_GAN, automatically selects the MLX variant when its
        metadata file is present and MLX is available.

        Args:
            expected: When provided, every key in this mapping is
                      compared against the metadata persisted at save
                      time.  A mismatch raises
                      ``GANMetadataMismatchError`` with a per-key diff
                      so the caller can decide whether to retrain the
                      GAN or update the strategy.  Common use:

                          interface.load(expected={
                              "min_buy_gain_threshold": self.MIN_BUY_GAIN_THRESHOLD,
                              "min_sell_loss_threshold": self.MIN_SELL_LOSS_THRESHOLD,
                              "training_type": int(self.TRAINING_TYPE),
                              "num_features": expected_features,
                              "column_order": list(train_df.columns),
                          })

                      Keys absent from the saved metadata are reported
                      as a mismatch (saved=<MISSING>) — stricter than
                      "best-effort", which is the point.

        Returns:
            Metadata dict stored alongside the model (thresholds,
            feature stats, etc.).  Returned even when ``expected`` is
            given so callers can inspect additional saved fields.
        """
        # Phase 2: migrated types use the GANBackend registry's
        # MLX-then-TF fallback probe.
        if self.gan_type in _BACKEND_MIGRATED:
            backend, metadata = load_with_fallback(
                self.gan_type, self.save_path, prefer_mlx=self._use_mlx
            )
            self._backend = backend
            self._model = getattr(backend, "_model", None)
            metadata_dict = dict(metadata or {})
            if expected:
                _validate_metadata_against_expected(
                    saved=metadata_dict,
                    expected=expected,
                    gan_type=self.gan_type,
                    save_path=self.save_path,
                )
            return metadata_dict

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


# ---------------------------------------------------------------------------
# Internal: metadata validation
# ---------------------------------------------------------------------------


_MISSING = object()  # sentinel for "key not present in saved metadata"


def _values_match(saved: Any, expected: Any) -> bool:
    """Compare a saved metadata value against an expected value.

    Numbers are compared with ``math.isclose`` to absorb pickle's
    np.float32→Python-float drift; sequences (lists/tuples/np arrays)
    are compared element-wise after converting to a common form;
    everything else falls back to ``==``.
    """
    if saved is _MISSING:
        return False

    # Numpy scalars — coerce to Python types for clean comparison.
    if isinstance(saved, np.generic):
        saved = saved.item()
    if isinstance(expected, np.generic):
        expected = expected.item()

    # Float-like comparison handles ``np.float32(0.016) ≈ 0.016``.
    if isinstance(saved, float) or isinstance(expected, float):
        try:
            return math.isclose(float(saved), float(expected), rel_tol=1e-6, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False

    # Sequences — order matters for column_order, so we don't sort.
    saved_is_seq = isinstance(saved, (list, tuple, np.ndarray))
    expected_is_seq = isinstance(expected, (list, tuple, np.ndarray))
    if saved_is_seq or expected_is_seq:
        if not (saved_is_seq and expected_is_seq):
            return False
        saved_list = list(saved)
        expected_list = list(expected)
        if len(saved_list) != len(expected_list):
            return False
        return all(_values_match(s, e) for s, e in zip(saved_list, expected_list))

    # Dicts (e.g. task_label_dims) — recursive value compare on shared keys.
    if isinstance(saved, dict) and isinstance(expected, dict):
        if set(saved.keys()) != set(expected.keys()):
            return False
        return all(_values_match(saved[k], expected[k]) for k in expected)

    return saved == expected


def _validate_metadata_against_expected(
    *,
    saved: Mapping[str, Any],
    expected: Mapping[str, Any],
    gan_type: GANType,
    save_path: str,
) -> None:
    """Raise ``GANMetadataMismatchError`` if any expected key is missing
    or unequal in the persisted metadata.

    Strict by design: a mismatch means the GAN was trained against a
    different config than the one currently active, and continuing
    would silently corrupt training (e.g. labels generated under one
    threshold combined with a GAN trained under another).  Either the
    GAN must be retrained or the strategy must be updated to match —
    we refuse to choose silently.
    """
    mismatches: Dict[str, Any] = {}
    for key, expected_value in expected.items():
        saved_value = saved.get(key, _MISSING)
        if not _values_match(saved_value, expected_value):
            display_saved = "<MISSING>" if saved_value is _MISSING else saved_value
            mismatches[key] = (display_saved, expected_value)

    if mismatches:
        raise GANMetadataMismatchError(
            mismatches=mismatches,
            gan_type=gan_type,
            save_path=save_path,
        )
