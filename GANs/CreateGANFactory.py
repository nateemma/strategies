# pragma pylint: disable=C0114, C0115, C0116
"""
Factory helpers for GAN creator strategies and GANInterface instances.

Two entry points are provided:

CreateGANFactory.create(key, ...)
    Instantiates a *creator strategy* (a Create* class that runs as a
    Freqtrade strategy to train and save a GAN model).

CreateGANFactory.interface(gan_type, save_path)
    Returns a GANInterface configured for the given GANType.  Use this
    from within strategies that need to augment training data at runtime.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Tuple

from GANs.GANType import GANType


class CreateGANFactory:
    """Lightweight factory for GAN creator strategies and inference interfaces.

    The creator registry maps short keys to (module_path, class_name).
    New types can be added at runtime via register().
    """

    # Creator strategies are now unified into CreateGAN (single-task) and
    # CreateMTGAN (multi-task); the specific backend is selected via each
    # class's gan_type attribute.  The legacy per-backend names continue
    # to resolve via thin compat subclasses so existing freqtrade configs
    # (``--strategy CreateWGAN`` etc.) keep working.
    _REGISTRY: Dict[str, Tuple[str, str]] = {
        "gan":         ("user_data.strategies.GANs.CreateGAN",           "CreateGAN"),
        "mt_gan":      ("user_data.strategies.GANs.CreateMTGAN",         "CreateMTGAN"),
        "wgan":        ("user_data.strategies.GANs.CreateWGAN",          "CreateWGAN"),
        "mt_wgan":     ("user_data.strategies.GANs.CreateMTWGAN",        "CreateMTWGAN"),
        "ctab_gan":    ("user_data.strategies.GANs.CreateCtabGanPlus",   "CreateCtabGanPlus"),
        "mt_ctab_gan": ("user_data.strategies.GANs.CreateMTCtabGanPlus", "CreateMTCtabGanPlus"),
        "cgp_pca":     ("user_data.strategies.GANs.Create_CGP_PCA",      "Create_CGP_PCA"),
    }

    # Maps GANType members to creator registry keys.  Uses the legacy
    # backend-specific shim names so each GANType still resolves to a
    # creator with the correct gan_type preset.
    _GANTYPE_TO_KEY: Dict[str, str] = {
        GANType.WGAN.name:        "wgan",
        GANType.MT_WGAN.name:     "mt_wgan",
        GANType.CTAB_GAN.name:    "ctab_gan",
        GANType.MT_CTAB_GAN.name: "mt_ctab_gan",
    }

    # ------------------------------------------------------------------ #
    # Creator strategy API                                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def register(cls, key: str, module_path: str, class_name: str) -> None:
        """Register a new creator strategy under key."""
        cls._REGISTRY[key] = (module_path, class_name)

    @classmethod
    def available(cls) -> Tuple[str, ...]:
        """Return all registered creator keys."""
        return tuple(cls._REGISTRY.keys())

    @classmethod
    def create(
        cls,
        key: str,
        gan_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Instantiate a creator strategy by registry key.

        Args:
            key:        One of the keys returned by available().
            gan_config: Configuration dict merged into DEFAULT_GAN_CONFIG.
            **kwargs:   Forwarded to the creator's __init__.
        """
        if key not in cls._REGISTRY:
            available = ", ".join(cls.available())
            raise ValueError(
                f"Unknown GAN creator key '{key}'. Available: {available}"
            )
        module_path, class_name = cls._REGISTRY[key]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)(gan_config=gan_config, **kwargs)

    # ------------------------------------------------------------------ #
    # Interface API                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def interface(
        cls,
        gan_type: GANType,
        save_path: str,
        prefer_mlx: bool = True,
    ) -> "GANInterface":  # type: ignore[name-defined]
        """Return a GANInterface configured for gan_type and save_path.

        Example::

            from GANs.GANType import GANType
            from GANs.CreateGANFactory import CreateGANFactory

            iface = CreateGANFactory.interface(GANType.WGAN, save_path="...")
            aug_x, aug_y = iface.balance(data, labels)

        Args:
            gan_type:   GAN backend to use.
            save_path:  Directory for model files.
            prefer_mlx: Prefer MLX when available (default True).
        """
        from GANs.GANInterface import GANInterface  # local import avoids circulars
        return GANInterface(gan_type=gan_type, save_path=save_path, prefer_mlx=prefer_mlx)

    @classmethod
    def creator_key_for(cls, gan_type: GANType) -> Optional[str]:
        """Return the creator registry key for a GANType, or None if not mapped."""
        return cls._GANTYPE_TO_KEY.get(gan_type.name)
