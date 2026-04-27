"""
GANBackend — abstract base class and registry for GAN backend implementations.

Each concrete backend wraps a single trainer/model class (CTABGANPlus,
WGANMLX, MTWGANGP, etc.) and exposes a uniform fit / generate / save /
load API.  GANInterface picks a backend by (GANType, prefer_mlx) and
delegates uniformly, so the heterogeneity that used to live as
``if self.gan_type == GANType.WGAN: ...`` branches in GANInterface is
now contained inside each backend.

Phase 2 of the GAN refactor — see commit history for context.

Usage:
    @register_backend
    class MyBackend(GANBackend):
        GAN_TYPE = GANType.WGAN
        PREFERS_MLX = False

        @classmethod
        def is_available(cls) -> bool:
            try:
                import tensorflow  # noqa
                return True
            except ImportError:
                return False

        def fit(self, data, labels, **kwargs): ...
        def generate(self, n, **kwargs): ...
        def save(self, save_path, **extra_metadata): ...

        @classmethod
        def load(cls, save_path):
            ...
            return instance, metadata

    # Caller (typically GANInterface):
    backend_cls = resolve_backend(GANType.WGAN, prefer_mlx=True)
    backend = backend_cls()
    backend.fit(data, labels)
    backend.save("/tmp/model")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

from GANs.GANType import GANType


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class GANBackend(ABC):
    """
    Abstract base class for GAN backends.

    Concrete subclasses must:
      * Set ``GAN_TYPE`` to the GANType they implement.
      * Set ``PREFERS_MLX`` to True for MLX-native backends, False otherwise.
      * Implement fit / generate / save / load.
      * Optionally override is_available() to gate registration on the
        availability of an underlying dependency (TF, MLX, etc.).
    """

    GAN_TYPE: ClassVar[GANType]
    PREFERS_MLX: ClassVar[bool] = False

    # ---------- lifecycle ---------- #

    @abstractmethod
    def fit(
        self,
        data: Any,
        labels: Any,
        categorical_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Train the GAN on ``(data, labels)``.

        Args:
            data:                Training data.  Shape and type vary by backend
                                 (numpy 2-D for tabular WGAN, 3-D for sequential
                                 WGAN/CGAN, DataFrame for CTAB family).
            labels:              One-hot ``np.ndarray`` for single-task backends
                                 or ``Dict[str, np.ndarray]`` for multi-task.
            categorical_columns: Used by CTAB family; ignored elsewhere.
            **kwargs:            Backend-specific hyperparameters (epochs,
                                 batch_size, learning_rate, etc.).  Backends
                                 should silently ignore keys they don't use
                                 so callers can pass a single config dict.
        """

    @abstractmethod
    def generate(self, n: int, **kwargs: Any) -> Any:
        """Generate ``n`` samples conditioned on backend-specific kwargs.

        Conditioning conventions (preserved from the pre-refactor API):
          * WGAN, CGAN          — ``one_hot=<np.ndarray (n, num_classes)>``
          * MT_WGAN, MT_CTAB_GAN — ``task_labels=<dict[str, np.ndarray]>``
          * CTAB_GAN            — ``class_label=<int>``

        Returns:
            Backend-specific output (np.ndarray for sequential backends,
            DataFrame for CTAB-family, tuple of (samples, labels) for
            multi-task).
        """

    @abstractmethod
    def save(self, save_path: str, **extra_metadata: Any) -> None:
        """Persist the trained model to ``save_path``.

        ``extra_metadata`` is merged into whatever the backend writes to
        its metadata file.  Typical callers pass MASTER threshold values
        (min_buy_gain_threshold, min_sell_loss_threshold, training_type)
        from the strategy that trained the GAN.
        """

    @classmethod
    @abstractmethod
    def load(cls, save_path: str) -> Tuple["GANBackend", Dict[str, Any]]:
        """Reconstruct a backend from disk.

        Returns a ``(backend_instance, metadata_dict)`` tuple — the backend
        is fitted and ready for ``generate()``, the metadata is whatever
        was persisted at save time (used by callers to recover MASTER
        thresholds and feature stats).
        """

    # ---------- capability gating ---------- #

    @classmethod
    def is_available(cls) -> bool:
        """Whether this backend can be constructed in the current environment.

        Default: always available.  MLX backends override this to check
        for ``mlx.core.metal.is_available()``; TF backends check for the
        ``tensorflow`` import.
        """
        return True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# Stored as: GANType → list of (PREFERS_MLX, backend_class) tuples.
# The list is in registration order; lookup respects prefer_mlx by trying
# all matching MLX entries first, then non-MLX entries.
_BACKEND_REGISTRY: Dict[GANType, List[Tuple[bool, Type[GANBackend]]]] = defaultdict(list)


def register_backend(cls: Type[GANBackend]) -> Type[GANBackend]:
    """
    Decorator: add a backend class to the global registry.

    The class must define ``GAN_TYPE`` and ``PREFERS_MLX`` before this
    decorator runs (i.e. as class-level attributes, not via __init__).
    """
    if not hasattr(cls, "GAN_TYPE"):
        raise TypeError(
            f"{cls.__name__} must define class attribute GAN_TYPE before registration"
        )
    _BACKEND_REGISTRY[cls.GAN_TYPE].append((cls.PREFERS_MLX, cls))
    return cls


def resolve_backend(gan_type: GANType, prefer_mlx: bool = False) -> Type[GANBackend]:
    """
    Pick the best available backend for ``(gan_type, prefer_mlx)``.

    Order of preference:
      1. If ``prefer_mlx`` is True, try MLX backends (in registration order)
         and return the first available one.
      2. Otherwise, or if no MLX backend is available, try non-MLX backends.
      3. If nothing is available, raise ``ValueError`` with a diagnostic
         listing every registered backend for that type.
    """
    candidates = _BACKEND_REGISTRY.get(gan_type, [])

    if prefer_mlx:
        for is_mlx, cls in candidates:
            if is_mlx and cls.is_available():
                return cls

    for is_mlx, cls in candidates:
        if not is_mlx and cls.is_available():
            return cls

    raise ValueError(
        f"No available backend for GANType.{gan_type.name} "
        f"(prefer_mlx={prefer_mlx}). "
        f"Registered backends: "
        f"{[(m, c.__name__) for m, c in candidates]}"
    )


def list_backends(
    gan_type: Optional[GANType] = None,
) -> List[Type[GANBackend]]:
    """List registered backends, optionally filtered by ``gan_type``."""
    if gan_type is not None:
        return [cls for _, cls in _BACKEND_REGISTRY.get(gan_type, [])]
    return [
        cls
        for entries in _BACKEND_REGISTRY.values()
        for _, cls in entries
    ]


def fit_with_fallback(
    gan_type: GANType,
    data: Any,
    labels: Any,
    *,
    prefer_mlx: bool = False,
    categorical_columns: Optional[List[str]] = None,
    **kwargs: Any,
) -> GANBackend:
    """
    Try every available backend for ``gan_type`` until one successfully
    fits.  Same MLX-then-TF fallback semantics as ``load_with_fallback``
    — useful when an MLX backend's underlying module fails to import
    (e.g. partial install) and we want to gracefully fall through to TF.

    Only ``ImportError`` / ``ModuleNotFoundError`` from a backend's fit
    are treated as "try the next one"; other exceptions propagate.

    Returns the fitted backend instance.
    """
    candidates = _BACKEND_REGISTRY.get(gan_type, [])
    last_error: Optional[Exception] = None

    def _try(is_mlx_filter):
        nonlocal last_error
        for is_mlx, cls in candidates:
            if is_mlx_filter(is_mlx) or not cls.is_available():
                continue
            try:
                instance = cls()
                instance.fit(
                    data, labels,
                    categorical_columns=categorical_columns,
                    **kwargs,
                )
                return instance
            except (ImportError, ModuleNotFoundError) as e:
                last_error = e
                continue
        return None

    if prefer_mlx:
        instance = _try(lambda is_mlx: not is_mlx)  # MLX backends only
        if instance is not None:
            return instance

    instance = _try(lambda is_mlx: is_mlx)  # non-MLX backends only
    if instance is not None:
        return instance

    raise RuntimeError(
        f"No backend could fit GANType.{gan_type.name} "
        f"(prefer_mlx={prefer_mlx}). Last error: {last_error}"
    )


def load_with_fallback(
    gan_type: GANType,
    save_path: str,
    prefer_mlx: bool = False,
) -> Tuple[GANBackend, Dict[str, Any]]:
    """
    Try every available backend for ``gan_type`` until one successfully loads.

    Order:
      1. If ``prefer_mlx`` is True, try MLX backends in registration order.
         Each is given a chance to ``load(save_path)``; ``FileNotFoundError``
         or import-failure means "not my format" and we move on.
      2. Then try non-MLX backends with the same fallback semantics.
      3. If nothing succeeds, raise ``RuntimeError`` with the last error
         seen so the caller can diagnose.

    Mirrors the manual MLX-then-TF probe-and-fallback that GANInterface
    used to do inline for the CTAB family.
    """
    candidates = _BACKEND_REGISTRY.get(gan_type, [])
    last_error: Optional[Exception] = None

    if prefer_mlx:
        for is_mlx, cls in candidates:
            if not is_mlx or not cls.is_available():
                continue
            try:
                return cls.load(save_path)
            except (FileNotFoundError, ImportError, ModuleNotFoundError) as e:
                last_error = e
                continue

    for is_mlx, cls in candidates:
        if is_mlx or not cls.is_available():
            continue
        try:
            return cls.load(save_path)
        except (FileNotFoundError, ImportError, ModuleNotFoundError) as e:
            last_error = e
            continue

    raise RuntimeError(
        f"No backend could load {save_path} for GANType.{gan_type.name} "
        f"(prefer_mlx={prefer_mlx}). "
        f"Last error: {last_error}"
    )


def clear_registry() -> None:
    """Remove every registered backend.  Test-only helper."""
    _BACKEND_REGISTRY.clear()
